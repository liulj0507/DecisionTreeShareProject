/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.tree.impl

import java.io.IOException

import scala.collection.mutable
import scala.util.Random

import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.tree._
import org.apache.spark.ml.util.Instrumentation
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo, Strategy => OldStrategy}
import org.apache.spark.mllib.tree.impurity.ImpurityCalculator
import org.apache.spark.mllib.tree.model.ImpurityStats
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.random.{SamplingUtils, XORShiftRandom}


/**
 * ALGORITHM
 *
 *这个算法按照行对样本进行分割
 *
 *每一次迭代中，算法切分结点集合，为了在每次迭代中选择最佳的切分点，会在分布式数据中收集足够的统计，
 *对于每一个结点，一些工作结点用于统计，并选择出最佳的切分点
 *
 * This setup requires discretization of continuous features.  This binning is done in the
 * findSplits() method during initialization, after which each continuous feature becomes
 * an ordered discretized feature with at most maxBins possible values.
 *
 * The main loop in the algorithm operates on a queue of nodes (nodeQueue).  These nodes
 * lie at the periphery of the tree being trained.  If multiple trees are being trained at once,
 * then this queue contains nodes from all of them.  Each iteration works roughly as follows:
 *   On the master node:
 *     - Some number of nodes are pulled off of the queue (based on the amount of memory
 *       required for their sufficient statistics).
 *     - For random forests, if featureSubsetStrategy is not "all," then a subset of candidate
 *       features are chosen for each node.  See method selectNodesToSplit().
 *   On worker nodes, via method findBestSplits():
 *     - The worker makes one pass over its subset of instances.
 *     - For each (tree, node, feature, split) tuple, the worker collects statistics about
 *       splitting.  Note that the set of (tree, node) pairs is limited to the nodes selected
 *       from the queue for this iteration.  The set of features considered can also be limited
 *       based on featureSubsetStrategy.
 *     - For each node, the statistics for that node are aggregated to a particular worker
 *       via reduceByKey().  The designated worker chooses the best (feature, split) pair,
 *       or chooses to stop splitting if the stopping criteria are met.
 *   On the master node:
 *     - The master collects all decisions about splitting nodes and updates the model.
 *     - The updated model is passed to the workers on the next iteration.
 * This process continues until the node queue is empty.
 *
 * Most of the methods in this implementation support the statistics aggregation, which is
 * the heaviest part of the computation.  In general, this implementation is bound by either
 * the cost of statistics computation on workers or by communicating the sufficient statistics.
 */
private[spark] object RandomForest extends Logging {

  /**
   * 训练随机森林
   *
   * @param input Training data: RDD of [[org.apache.spark.mllib.regression.LabeledPoint]]
   * @return an unweighted set of trees
   */
  def run(
      input: RDD[LabeledPoint],
      strategy: OldStrategy,
      numTrees: Int,
      featureSubsetStrategy: String,
      seed: Long,
      instr: Option[Instrumentation[_]],
      parentUID: Option[String] = None): Array[DecisionTreeModel] = {

    val timer = new TimeTracker()

    timer.start("total")

    timer.start("init")

    val retaggedInput = input.retag(classOf[LabeledPoint])
    // 建立决策树的元数据信息（包括分裂点位置、箱子数及各箱子包含特征属性的值等）
    val metadata =
      DecisionTreeMetadata.buildMetadata(retaggedInput, strategy, numTrees, featureSubsetStrategy)
    instr match {
      case Some(instrumentation) =>
        instrumentation.logNumFeatures(metadata.numFeatures)
        instrumentation.logNumClasses(metadata.numClasses)
      case None =>
        logInfo("numFeatures: " + metadata.numFeatures)
        logInfo("numClasses: " + metadata.numClasses)
    }

    timer.start("findSplits")
    // 找到切分点(splits)和箱子信息(bins,切分点之间的间隔)
    // 对于连续型特征，利用切分点抽样统计简化计算
    // 对于离散型特征，如果是无序的，则最多有splits = 2 ^ (numBins - 1)个划分
    // 对于离散型特征，如果是有序的，则最多有splits = numBins - 1个划分
    val splits = findSplits(retaggedInput, metadata, seed)
    timer.stop("findSplits")
    logDebug("numBins: feature: number of bins")
    logDebug(Range(0, metadata.numFeatures).map { featureIndex =>
      s"\t$featureIndex\t${metadata.numBins(featureIndex)}"
    }.mkString("\n"))

    // Bin feature values (TreePoint representation).
    // Cache input RDD for speedup during multiple passes.
    // 转换成树形的RDD类型，转换后，所有样本点已经按分裂点条件分到了各自的箱子中
    val treeInput = TreePoint.convertToTreeRDD(retaggedInput, splits, metadata)

    val withReplacement = numTrees > 1

    // convertToBaggedRDD 使得每棵树就是样本的一个子集
    val baggedInput = BaggedPoint
      .convertToBaggedRDD(treeInput, strategy.subsamplingRate, numTrees, withReplacement, seed)
      .persist(StorageLevel.MEMORY_AND_DISK)

    // depth of the decision tree
    // 决策树的深度
    val maxDepth = strategy.maxDepth
    require(maxDepth <= 30,
      s"DecisionTree currently only supports maxDepth <= 30, but was given maxDepth = $maxDepth.")

    // Max memory usage for aggregates
    // TODO: Calculate memory usage more precisely.
    val maxMemoryUsage: Long = strategy.maxMemoryInMB * 1024L * 1024L
    logDebug("max memory usage for aggregates = " + maxMemoryUsage + " bytes.")

    /*
     * The main idea here is to perform group-wise training of the decision tree nodes thus
     * reducing the passes over the data from (# nodes) to (# nodes / maxNumberOfNodesPerGroup).
     * Each data sample is handled by a particular node (or it reaches a leaf and is not used
     * in lower levels).
     */

    // Create an RDD of node Id cache.
    // At first, all the rows belong to the root nodes (node Id == 1).
    // 结点是否使用缓存，结点ID从1开始，1就是这棵树的根结点，左结点为2,右结点为3,依次递增
    val nodeIdCache = if (strategy.useNodeIdCache) {
      Some(NodeIdCache.init(
        data = baggedInput,
        numTrees = numTrees,
        checkpointInterval = strategy.checkpointInterval,
        initVal = 1))
    } else {
      None
    }

    // FIFO queue of nodes to train: (treeIndex, node)
    val nodeQueue = new mutable.Queue[(Int, LearningNode)]()

    val rng = new Random()
    rng.setSeed(seed)

    // Allocate and queue root nodes.
    // 创建树的根结点
    val topNodes = Array.fill[LearningNode](numTrees)(LearningNode.emptyNode(nodeIndex = 1))
    // 将(树的索引，树的根结点)这个tuple压入队列，树索引从0开始，根结点从1开始
    Range(0, numTrees).foreach(treeIndex => nodeQueue.enqueue((treeIndex, topNodes(treeIndex))))

    timer.stop("init")

    while (nodeQueue.nonEmpty) {
      // Collect some nodes to split, and choose features for each node (if subsampling).
      // Each group of nodes may come from one or multiple trees, and at multiple levels.
      // 取得每棵树所有需要切分的结点
      val (nodesForGroup, treeToNodeToIndexInfo) =
        RandomForest.selectNodesToSplit(nodeQueue, maxMemoryUsage, metadata, rng)
      // Sanity check (should never occur):
      assert(nodesForGroup.nonEmpty,
        s"RandomForest selected empty nodesForGroup.  Error for unknown reason.")

      // Choose node splits, and enqueue new nodes as needed.
      timer.start("findBestSplits")
      // 找出最优切分点
      RandomForest.findBestSplits(baggedInput, metadata, topNodes, nodesForGroup,
        treeToNodeToIndexInfo, splits, nodeQueue, timer, nodeIdCache)
      timer.stop("findBestSplits")
    }

    baggedInput.unpersist()

    timer.stop("total")

    logInfo("Internal timing for DecisionTree:")
    logInfo(s"$timer")

    // Delete any remaining checkpoints used for node Id cache.
    if (nodeIdCache.nonEmpty) {
      try {
        nodeIdCache.get.deleteAllCheckpoints()
      } catch {
        case e: IOException =>
          logWarning(s"delete all checkpoints failed. Error reason: ${e.getMessage}")
      }
    }

    val numFeatures = metadata.numFeatures

    parentUID match {
      case Some(uid) =>
        if (strategy.algo == OldAlgo.Classification) {
          topNodes.map { rootNode =>
            new DecisionTreeClassificationModel(uid, rootNode.toNode, numFeatures,
              strategy.getNumClasses)
          }
        } else {
          topNodes.map { rootNode =>
            new DecisionTreeRegressionModel(uid, rootNode.toNode, numFeatures)
          }
        }
      case None =>
        if (strategy.algo == OldAlgo.Classification) {
          topNodes.map { rootNode =>
            new DecisionTreeClassificationModel(rootNode.toNode, numFeatures,
              strategy.getNumClasses)
          }
        } else {
          topNodes.map(rootNode => new DecisionTreeRegressionModel(rootNode.toNode, numFeatures))
        }
    }
  }

  /**
   * Helper for binSeqOp, for data which can contain a mix of ordered and unordered features.
   *
   * For ordered features, a single bin is updated.
   * For unordered features, bins correspond to subsets of categories; either the left or right bin
   * for each subset is updated.
   *
   * @param agg  Array storing aggregate calculation, with a set of sufficient statistics for
   *             each (feature, bin).
   * @param treePoint  Data point being aggregated.
   * @param splits possible splits indexed (numFeatures)(numSplits)
   * @param unorderedFeatures  Set of indices of unordered features.
   * @param instanceWeight  Weight (importance) of instance in dataset.
   */
  private def mixedBinSeqOp(
      agg: DTStatsAggregator,
      treePoint: TreePoint,
      splits: Array[Array[Split]],
      unorderedFeatures: Set[Int],
      instanceWeight: Double,
      featuresForNode: Option[Array[Int]]): Unit = {
    val numFeaturesPerNode = if (featuresForNode.nonEmpty) {
      // Use subsampled features
      featuresForNode.get.length
    } else {
      // Use all features
      agg.metadata.numFeatures
    }
    // Iterate over features.
    var featureIndexIdx = 0
    while (featureIndexIdx < numFeaturesPerNode) {
      val featureIndex = if (featuresForNode.nonEmpty) {
        featuresForNode.get.apply(featureIndexIdx)
      } else {
        featureIndexIdx
      }
      if (unorderedFeatures.contains(featureIndex)) {
        // Unordered feature
        val featureValue = treePoint.binnedFeatures(featureIndex)
        val leftNodeFeatureOffset = agg.getFeatureOffset(featureIndexIdx)
        // Update the left or right bin for each split.
        val numSplits = agg.metadata.numSplits(featureIndex)
        val featureSplits = splits(featureIndex)
        var splitIndex = 0
        while (splitIndex < numSplits) {
          if (featureSplits(splitIndex).shouldGoLeft(featureValue, featureSplits)) {
            agg.featureUpdate(leftNodeFeatureOffset, splitIndex, treePoint.label, instanceWeight)
          }
          splitIndex += 1
        }
      } else {
        // Ordered feature
        val binIndex = treePoint.binnedFeatures(featureIndex)
        agg.update(featureIndexIdx, binIndex, treePoint.label, instanceWeight)
      }
      featureIndexIdx += 1
    }
  }

  /**
   * Helper for binSeqOp, for regression and for classification with only ordered features.
   *
   * For each feature, the sufficient statistics of one bin are updated.
   *
   * @param agg  Array storing aggregate calculation, with a set of sufficient statistics for
   *             each (feature, bin).
   * @param treePoint  Data point being aggregated.
   * @param instanceWeight  Weight (importance) of instance in dataset.
   */
  private def orderedBinSeqOp(
      agg: DTStatsAggregator,
      treePoint: TreePoint,
      instanceWeight: Double,
      featuresForNode: Option[Array[Int]]): Unit = {
    val label = treePoint.label

    // Iterate over features.
    if (featuresForNode.nonEmpty) {
      // Use subsampled features
      var featureIndexIdx = 0
      while (featureIndexIdx < featuresForNode.get.length) {
        val binIndex = treePoint.binnedFeatures(featuresForNode.get.apply(featureIndexIdx))
        agg.update(featureIndexIdx, binIndex, label, instanceWeight)
        featureIndexIdx += 1
      }
    } else {
      // Use all features
      val numFeatures = agg.metadata.numFeatures
      var featureIndex = 0
      while (featureIndex < numFeatures) {
        val binIndex = treePoint.binnedFeatures(featureIndex)
        agg.update(featureIndex, binIndex, label, instanceWeight)
        featureIndex += 1
      }
    }
  }

  /**
   * 给定一组结点，找出每个结点的最优切分点
   *
   * @param input Training data: RDD of [[org.apache.spark.ml.tree.impl.TreePoint]]
   * @param metadata Learning and dataset metadata
   * @param topNodes Root node for each tree.  Used for matching instances with nodes.
   * @param nodesForGroup Mapping: treeIndex --> nodes to be split in tree
   * @param treeToNodeToIndexInfo Mapping: treeIndex --> nodeIndex --> nodeIndexInfo,
   *                              where nodeIndexInfo stores the index in the group and the
   *                              feature subsets (if using feature subsets).
   * @param splits possible splits for all features, indexed (numFeatures)(numSplits)
   * @param nodeQueue  Queue of nodes to split, with values (treeIndex, node).
   *                   Updated with new non-leaf nodes which are created.
   * @param nodeIdCache Node Id cache containing an RDD of Array[Int] where
   *                    each value in the array is the data point's node Id
   *                    for a corresponding tree. This is used to prevent the need
   *                    to pass the entire tree to the executors during
   *                    the node stat aggregation phase.
   */
  private[tree] def findBestSplits(
      input: RDD[BaggedPoint[TreePoint]],
      metadata: DecisionTreeMetadata,
      topNodes: Array[LearningNode],
      nodesForGroup: Map[Int, Array[LearningNode]],
      treeToNodeToIndexInfo: Map[Int, Map[Int, NodeIndexInfo]],
      splits: Array[Array[Split]],
      nodeQueue: mutable.Queue[(Int, LearningNode)],
      timer: TimeTracker = new TimeTracker,
      nodeIdCache: Option[NodeIdCache] = None): Unit = {

    /*
     * The high-level descriptions of the best split optimizations are noted here.
     *
     * *Group-wise training*
     * We perform bin calculations for groups of nodes to reduce the number of
     * passes over the data.  Each iteration requires more computation and storage,
     * but saves several iterations over the data.
     *
     * *Bin-wise computation*
     * We use a bin-wise best split computation strategy instead of a straightforward best split
     * computation strategy. Instead of analyzing each sample for contribution to the left/right
     * child node impurity of every split, we first categorize each feature of a sample into a
     * bin. We exploit this structure to calculate aggregates for bins and then use these aggregates
     * to calculate information gain for each split.
     *
     * *Aggregation over partitions*
     * Instead of performing a flatMap/reduceByKey operation, we exploit the fact that we know
     * the number of splits in advance. Thus, we store the aggregates (at the appropriate
     * indices) in a single array for all bins and rely upon the RDD aggregate method to
     * drastically reduce the communication overhead.
     */

    // numNodes:  Number of nodes in this group
    val numNodes = nodesForGroup.values.map(_.length).sum
    logDebug("numNodes = " + numNodes)
    logDebug("numFeatures = " + metadata.numFeatures)
    logDebug("numClasses = " + metadata.numClasses)
    logDebug("isMulticlass = " + metadata.isMulticlass)
    logDebug("isMulticlassWithCategoricalFeatures = " +
      metadata.isMulticlassWithCategoricalFeatures)
    logDebug("using nodeIdCache = " + nodeIdCache.nonEmpty.toString)

    /**
     * Performs a sequential aggregation over a partition for a particular tree and node.
     *
     * For each feature, the aggregate sufficient statistics are updated for the relevant
     * bins.
     *
     * @param treeIndex Index of the tree that we want to perform aggregation for.
     * @param nodeInfo The node info for the tree node.
     * @param agg Array storing aggregate calculation, with a set of sufficient statistics
     *            for each (node, feature, bin).
     * @param baggedPoint Data point being aggregated.
     */
    def nodeBinSeqOp(
        treeIndex: Int,
        nodeInfo: NodeIndexInfo,
        agg: Array[DTStatsAggregator],
        baggedPoint: BaggedPoint[TreePoint]): Unit = {
      if (nodeInfo != null) {
        val aggNodeIndex = nodeInfo.nodeIndexInGroup
        val featuresForNode = nodeInfo.featureSubset
        val instanceWeight = baggedPoint.subsampleWeights(treeIndex)
        if (metadata.unorderedFeatures.isEmpty) {
          orderedBinSeqOp(agg(aggNodeIndex), baggedPoint.datum, instanceWeight, featuresForNode)
        } else {
          mixedBinSeqOp(agg(aggNodeIndex), baggedPoint.datum, splits,
            metadata.unorderedFeatures, instanceWeight, featuresForNode)
        }
        agg(aggNodeIndex).updateParent(baggedPoint.datum.label, instanceWeight)
      }
    }

    /**
     * Performs a sequential aggregation over a partition.
     *
     * Each data point contributes to one node. For each feature,
     * the aggregate sufficient statistics are updated for the relevant bins.
     *
     * @param agg  Array storing aggregate calculation, with a set of sufficient statistics for
     *             each (node, feature, bin).
     * @param baggedPoint   Data point being aggregated.
     * @return  agg
     */
    def binSeqOp(
        agg: Array[DTStatsAggregator],
        baggedPoint: BaggedPoint[TreePoint]): Array[DTStatsAggregator] = {
      treeToNodeToIndexInfo.foreach { case (treeIndex, nodeIndexToInfo) =>
        val nodeIndex = topNodes(treeIndex).predictImpl(baggedPoint.datum.binnedFeatures, splits)
        nodeBinSeqOp(treeIndex, nodeIndexToInfo.getOrElse(nodeIndex, null), agg, baggedPoint)
      }
      agg
    }

    /**
     * Do the same thing as binSeqOp, but with nodeIdCache.
     */
    def binSeqOpWithNodeIdCache(
        agg: Array[DTStatsAggregator],
        dataPoint: (BaggedPoint[TreePoint], Array[Int])): Array[DTStatsAggregator] = {
      treeToNodeToIndexInfo.foreach { case (treeIndex, nodeIndexToInfo) =>
        val baggedPoint = dataPoint._1
        val nodeIdCache = dataPoint._2
        val nodeIndex = nodeIdCache(treeIndex)
        nodeBinSeqOp(treeIndex, nodeIndexToInfo.getOrElse(nodeIndex, null), agg, baggedPoint)
      }

      agg
    }

    /**
     * Get node index in group --> features indices map,
     * which is a short cut to find feature indices for a node given node index in group.
     */
    def getNodeToFeatures(
        treeToNodeToIndexInfo: Map[Int, Map[Int, NodeIndexInfo]]): Option[Map[Int, Array[Int]]] = {
      if (!metadata.subsamplingFeatures) {
        None
      } else {
        val mutableNodeToFeatures = new mutable.HashMap[Int, Array[Int]]()
        treeToNodeToIndexInfo.values.foreach { nodeIdToNodeInfo =>
          nodeIdToNodeInfo.values.foreach { nodeIndexInfo =>
            assert(nodeIndexInfo.featureSubset.isDefined)
            mutableNodeToFeatures(nodeIndexInfo.nodeIndexInGroup) = nodeIndexInfo.featureSubset.get
          }
        }
        Some(mutableNodeToFeatures.toMap)
      }
    }

    // array of nodes to train indexed by node index in group
    val nodes = new Array[LearningNode](numNodes)
    nodesForGroup.foreach { case (treeIndex, nodesForTree) =>
      nodesForTree.foreach { node =>
        nodes(treeToNodeToIndexInfo(treeIndex)(node.id).nodeIndexInGroup) = node
      }
    }

    // Calculate best splits for all nodes in the group
    timer.start("chooseSplits")

    // In each partition, iterate all instances and compute aggregate stats for each node,
    // yield a (nodeIndex, nodeAggregateStats) pair for each node.
    // After a `reduceByKey` operation,
    // stats of a node will be shuffled to a particular partition and be combined together,
    // then best splits for nodes are found there.
    // Finally, only best Splits for nodes are collected to driver to construct decision tree.
    val nodeToFeatures = getNodeToFeatures(treeToNodeToIndexInfo)
    val nodeToFeaturesBc = input.sparkContext.broadcast(nodeToFeatures)

    val partitionAggregates: RDD[(Int, DTStatsAggregator)] = if (nodeIdCache.nonEmpty) {
      input.zip(nodeIdCache.get.nodeIdsForInstances).mapPartitions { points =>
        // Construct a nodeStatsAggregators array to hold node aggregate stats,
        // each node will have a nodeStatsAggregator
        val nodeStatsAggregators = Array.tabulate(numNodes) { nodeIndex =>
          val featuresForNode = nodeToFeaturesBc.value.map { nodeToFeatures =>
            nodeToFeatures(nodeIndex)
          }
          new DTStatsAggregator(metadata, featuresForNode)
        }

        // iterator all instances in current partition and update aggregate stats
        points.foreach(binSeqOpWithNodeIdCache(nodeStatsAggregators, _))

        // transform nodeStatsAggregators array to (nodeIndex, nodeAggregateStats) pairs,
        // which can be combined with other partition using `reduceByKey`
        nodeStatsAggregators.view.zipWithIndex.map(_.swap).iterator
      }
    } else {
      input.mapPartitions { points =>
        // Construct a nodeStatsAggregators array to hold node aggregate stats,
        // each node will have a nodeStatsAggregator
        val nodeStatsAggregators = Array.tabulate(numNodes) { nodeIndex =>
          val featuresForNode = nodeToFeaturesBc.value.flatMap { nodeToFeatures =>
            Some(nodeToFeatures(nodeIndex))
          }
          new DTStatsAggregator(metadata, featuresForNode)
        }

        // iterator all instances in current partition and update aggregate stats
        points.foreach(binSeqOp(nodeStatsAggregators, _))

        // transform nodeStatsAggregators array to (nodeIndex, nodeAggregateStats) pairs,
        // which can be combined with other partition using `reduceByKey`
        nodeStatsAggregators.view.zipWithIndex.map(_.swap).iterator
      }
    }

    val nodeToBestSplits = partitionAggregates.reduceByKey((a, b) => a.merge(b)).map {
      case (nodeIndex, aggStats) =>
        val featuresForNode = nodeToFeaturesBc.value.flatMap { nodeToFeatures =>
          Some(nodeToFeatures(nodeIndex))
        }

        // find best split for each node
        // 对每一个结点查找最佳分裂位置
        val (split: Split, stats: ImpurityStats) =
          binsToBestSplit(aggStats, splits, featuresForNode, nodes(nodeIndex))
        (nodeIndex, (split, stats))
    }.collectAsMap()

    timer.stop("chooseSplits")

    val nodeIdUpdaters = if (nodeIdCache.nonEmpty) {
      Array.fill[mutable.Map[Int, NodeIndexUpdater]](
        metadata.numTrees)(mutable.Map[Int, NodeIndexUpdater]())
    } else {
      null
    }
    // Iterate over all nodes in this group.
    nodesForGroup.foreach { case (treeIndex, nodesForTree) =>
      nodesForTree.foreach { node =>
        val nodeIndex = node.id
        val nodeInfo = treeToNodeToIndexInfo(treeIndex)(nodeIndex)
        val aggNodeIndex = nodeInfo.nodeIndexInGroup
        val (split: Split, stats: ImpurityStats) =
          nodeToBestSplits(aggNodeIndex)
        logDebug("best split = " + split)

        // Extract info for this node.  Create children if not leaf.
        val isLeaf =
          (stats.gain <= 0) || (LearningNode.indexToLevel(nodeIndex) == metadata.maxDepth)
        node.isLeaf = isLeaf
        node.stats = stats
        logDebug("Node = " + node)

        if (!isLeaf) {
          node.split = Some(split)
          val childIsLeaf = (LearningNode.indexToLevel(nodeIndex) + 1) == metadata.maxDepth
          val leftChildIsLeaf = childIsLeaf || (stats.leftImpurity == 0.0)
          val rightChildIsLeaf = childIsLeaf || (stats.rightImpurity == 0.0)
          node.leftChild = Some(LearningNode(LearningNode.leftChildIndex(nodeIndex),
            leftChildIsLeaf, ImpurityStats.getEmptyImpurityStats(stats.leftImpurityCalculator)))
          node.rightChild = Some(LearningNode(LearningNode.rightChildIndex(nodeIndex),
            rightChildIsLeaf, ImpurityStats.getEmptyImpurityStats(stats.rightImpurityCalculator)))

          if (nodeIdCache.nonEmpty) {
            val nodeIndexUpdater = NodeIndexUpdater(
              split = split,
              nodeIndex = nodeIndex)
            nodeIdUpdaters(treeIndex).put(nodeIndex, nodeIndexUpdater)
          }

          // enqueue left child and right child if they are not leaves
          if (!leftChildIsLeaf) {
            nodeQueue.enqueue((treeIndex, node.leftChild.get))
          }
          if (!rightChildIsLeaf) {
            nodeQueue.enqueue((treeIndex, node.rightChild.get))
          }

          logDebug("leftChildIndex = " + node.leftChild.get.id +
            ", impurity = " + stats.leftImpurity)
          logDebug("rightChildIndex = " + node.rightChild.get.id +
            ", impurity = " + stats.rightImpurity)
        }
      }
    }

    if (nodeIdCache.nonEmpty) {
      // Update the cache if needed.
      nodeIdCache.get.updateNodeIndices(input, nodeIdUpdaters, splits)
    }
  }

  /**
   * Calculate the impurity statistics for a given (feature, split) based upon left/right
   * aggregates.
   *
   * @param stats the recycle impurity statistics for this feature's all splits,
   *              only 'impurity' and 'impurityCalculator' are valid between each iteration
   * @param leftImpurityCalculator left node aggregates for this (feature, split)
   * @param rightImpurityCalculator right node aggregate for this (feature, split)
   * @param metadata learning and dataset metadata for DecisionTree
   * @return Impurity statistics for this (feature, split)
   */
  private def calculateImpurityStats(
      stats: ImpurityStats,
      leftImpurityCalculator: ImpurityCalculator,
      rightImpurityCalculator: ImpurityCalculator,
      metadata: DecisionTreeMetadata): ImpurityStats = {

    val parentImpurityCalculator: ImpurityCalculator = if (stats == null) {
      leftImpurityCalculator.copy.add(rightImpurityCalculator)
    } else {
      stats.impurityCalculator
    }

    val impurity: Double = if (stats == null) {
      parentImpurityCalculator.calculate()
    } else {
      stats.impurity
    }

    val leftCount = leftImpurityCalculator.count
    val rightCount = rightImpurityCalculator.count

    val totalCount = leftCount + rightCount

    // If left child or right child doesn't satisfy minimum instances per node,
    // then this split is invalid, return invalid information gain stats.
    if ((leftCount < metadata.minInstancesPerNode) ||
      (rightCount < metadata.minInstancesPerNode)) {
      return ImpurityStats.getInvalidImpurityStats(parentImpurityCalculator)
    }

    val leftImpurity = leftImpurityCalculator.calculate() // Note: This equals 0 if count = 0
    val rightImpurity = rightImpurityCalculator.calculate()

    val leftWeight = leftCount / totalCount.toDouble
    val rightWeight = rightCount / totalCount.toDouble

    val gain = impurity - leftWeight * leftImpurity - rightWeight * rightImpurity

    // if information gain doesn't satisfy minimum information gain,
    // then this split is invalid, return invalid information gain stats.
    if (gain < metadata.minInfoGain) {
      return ImpurityStats.getInvalidImpurityStats(parentImpurityCalculator)
    }

    new ImpurityStats(gain, impurity, parentImpurityCalculator,
      leftImpurityCalculator, rightImpurityCalculator)
  }

  /**
   * 查找一个结点的最佳分裂位置
   *
   * @param binAggregates Bin statistics.
   * @return tuple for best split: (Split, information gain, prediction at node)
   */
  private[tree] def binsToBestSplit(
      binAggregates: DTStatsAggregator,
      splits: Array[Array[Split]],
      featuresForNode: Option[Array[Int]],
      node: LearningNode): (Split, ImpurityStats) = {

    // Calculate InformationGain and ImpurityStats if current node is top node
    // 计算根结点的预测值及纯度
    val level = LearningNode.indexToLevel(node.id)
    var gainAndImpurityStats: ImpurityStats = if (level == 0) {
      null
    } else {
      node.stats
    }

    // For each (feature, split), calculate the gain, and select the best (feature, split).
    // 对于每一个特征的(feature,split)，计算该分裂的增益，最后选择最佳的分裂(feature,split)
    val (bestSplit, bestSplitStats) =
      Range(0, binAggregates.metadata.numFeaturesPerNode).map { featureIndexIdx =>
        val featureIndex = if (featuresForNode.nonEmpty) {
          featuresForNode.get.apply(featureIndexIdx)
        } else {
          featureIndexIdx
        }
        val numSplits = binAggregates.metadata.numSplits(featureIndex)
        if (binAggregates.metadata.isContinuous(featureIndex)) {
          // Cumulative sum (scanLeft) of bin statistics.
          // Afterwards, binAggregates for a bin is the sum of aggregates for
          // that bin + all preceding bins.
          // 连续特征
          val nodeFeatureOffset = binAggregates.getFeatureOffset(featureIndexIdx)
          var splitIndex = 0
          while (splitIndex < numSplits) {
            binAggregates.mergeForFeature(nodeFeatureOffset, splitIndex + 1, splitIndex)
            splitIndex += 1
          }
          // Find best split.
          // 遍历查找最佳分裂位置
          val (bestFeatureSplitIndex, bestFeatureGainStats) =
            Range(0, numSplits).map { case splitIdx =>
              val leftChildStats = binAggregates.getImpurityCalculator(nodeFeatureOffset, splitIdx)
              val rightChildStats =
                binAggregates.getImpurityCalculator(nodeFeatureOffset, numSplits)
              rightChildStats.subtract(leftChildStats)
              gainAndImpurityStats = calculateImpurityStats(gainAndImpurityStats,
                leftChildStats, rightChildStats, binAggregates.metadata)
              (splitIdx, gainAndImpurityStats)
            }.maxBy(_._2.gain)
          (splits(featureIndex)(bestFeatureSplitIndex), bestFeatureGainStats)
        } else if (binAggregates.metadata.isUnordered(featureIndex)) {
          // Unordered categorical feature
          // 离散无序特征
          val leftChildOffset = binAggregates.getFeatureOffset(featureIndexIdx)
          // 遍历查找最佳分裂位置
          val (bestFeatureSplitIndex, bestFeatureGainStats) =
            Range(0, numSplits).map { splitIndex =>
              val leftChildStats = binAggregates.getImpurityCalculator(leftChildOffset, splitIndex)
              val rightChildStats = binAggregates.getParentImpurityCalculator()
                .subtract(leftChildStats)
              gainAndImpurityStats = calculateImpurityStats(gainAndImpurityStats,
                leftChildStats, rightChildStats, binAggregates.metadata)
              (splitIndex, gainAndImpurityStats)
            }.maxBy(_._2.gain)
          (splits(featureIndex)(bestFeatureSplitIndex), bestFeatureGainStats)
        } else {
          // Ordered categorical feature
          //离散有序特征
          val nodeFeatureOffset = binAggregates.getFeatureOffset(featureIndexIdx)
          val numCategories = binAggregates.metadata.numBins(featureIndex)

          /* Each bin is one category (feature value).
           * The bins are ordered based on centroidForCategories, and this ordering determines which
           * splits are considered.  (With K categories, we consider K - 1 possible splits.)
           *
           * centroidForCategories is a list: (category, centroid)
           */
          val centroidForCategories = Range(0, numCategories).map { case featureValue =>
            val categoryStats =
              binAggregates.getImpurityCalculator(nodeFeatureOffset, featureValue)
            val centroid = if (categoryStats.count != 0) {
              if (binAggregates.metadata.isMulticlass) {
                // multiclass classification
                // For categorical variables in multiclass classification,
                // the bins are ordered by the impurity of their corresponding labels.
                // 多元分类
                // bin的有序性计算，对应标签的纯度
                categoryStats.calculate()
              } else if (binAggregates.metadata.isClassification) {
                // binary classification
                // For categorical variables in binary classification,
                // the bins are ordered by the count of class 1.
                // 二分类
                categoryStats.stats(1)
              } else {
                // regression
                // For categorical variables in regression and binary classification,
                // the bins are ordered by the prediction.
                // 回归
                categoryStats.predict
              }
            } else {
              Double.MaxValue
            }
            (featureValue, centroid)
          }

          logDebug("Centroids for categorical variable: " + centroidForCategories.mkString(","))

          // bins sorted by centroids
          // 排序
          val categoriesSortedByCentroid = centroidForCategories.toList.sortBy(_._2)

          logDebug("Sorted centroids for categorical variable = " +
            categoriesSortedByCentroid.mkString(","))

          // Cumulative sum (scanLeft) of bin statistics.
          // Afterwards, binAggregates for a bin is the sum of aggregates for
          // that bin + all preceding bins.
          // bin求和
          var splitIndex = 0
          while (splitIndex < numSplits) {
            val currentCategory = categoriesSortedByCentroid(splitIndex)._1
            val nextCategory = categoriesSortedByCentroid(splitIndex + 1)._1
            binAggregates.mergeForFeature(nodeFeatureOffset, nextCategory, currentCategory)
            splitIndex += 1
          }
          // lastCategory = index of bin with total aggregates for this (node, feature)
          val lastCategory = categoriesSortedByCentroid.last._1
          // Find best split.
          // 查找最优分裂
          val (bestFeatureSplitIndex, bestFeatureGainStats) =
            Range(0, numSplits).map { splitIndex =>
              val featureValue = categoriesSortedByCentroid(splitIndex)._1
              val leftChildStats =
                binAggregates.getImpurityCalculator(nodeFeatureOffset, featureValue)
              val rightChildStats =
                binAggregates.getImpurityCalculator(nodeFeatureOffset, lastCategory)
              rightChildStats.subtract(leftChildStats)
              gainAndImpurityStats = calculateImpurityStats(gainAndImpurityStats,
                leftChildStats, rightChildStats, binAggregates.metadata)
              (splitIndex, gainAndImpurityStats)
            }.maxBy(_._2.gain)
          val categoriesForSplit =
            categoriesSortedByCentroid.map(_._1.toDouble).slice(0, bestFeatureSplitIndex + 1)
          val bestFeatureSplit =
            new CategoricalSplit(featureIndex, categoriesForSplit.toArray, numCategories)
          (bestFeatureSplit, bestFeatureGainStats)
        }
      }.maxBy(_._2.gain)

    (bestSplit, bestSplitStats)
  }

  /**
   * 返回决策树的split和bin，
   * 找出每个特征可能出现的split和相应的bin
   * 对于连续型和离散型特征，处理方式不同
   *
   * Continuous features:
   *   For each feature, there are numBins - 1 possible splits representing the possible binary
   *   decisions at each node in the tree.
   *   This finds locations (feature values) for splits using a subsample of the data.
   *
   * Categorical features:
   *   For each feature, there is 1 bin per split.
   *   Splits and bins are handled in 2 ways:
   *   (a) "unordered features"
   *       For multiclass classification with a low-arity feature
   *       (i.e., if isMulticlass && isSpaceSufficientForAllCategoricalSplits),
   *       the feature is split based on subsets of categories.
   *   (b) "ordered features"
   *       For regression and binary classification,
   *       and for multiclass classification with a high-arity feature,
   *       there is one bin per category.
   *
   * @param input Training data: RDD of [[org.apache.spark.mllib.regression.LabeledPoint]]
   * @param metadata Learning and dataset metadata
   * @param seed random seed
   * @return Splits, an Array of [[org.apache.spark.mllib.tree.model.Split]]
   *          of size (numFeatures, numSplits)
   */
  protected[tree] def findSplits(
      input: RDD[LabeledPoint],
      metadata: DecisionTreeMetadata,
      seed: Long): Array[Array[Split]] = {

    logDebug("isMulticlass = " + metadata.isMulticlass)

    // 特征数量
    val numFeatures = metadata.numFeatures

    // Sample the input only if there are continuous features.
    // 对于连续型特征，取值会比较多，因此需要做抽样
    val continuousFeatures = Range(0, numFeatures).filter(metadata.isContinuous)
    val sampledInput = if (continuousFeatures.nonEmpty) {//连续型特征
      // Calculate the number of samples for approximate quantile calculation.
      // 计算抽样样本数量，抽样数量远远大于maxBins
      val requiredSamples = math.max(metadata.maxBins * metadata.maxBins, 10000)
      // 计算抽样比例
      val fraction = if (requiredSamples < metadata.numExamples) {
        requiredSamples.toDouble / metadata.numExamples
      } else {
        1.0
      }
      logDebug("fraction of data used for calculating quantiles = " + fraction)
      // 对输入样本进行抽样
      input.sample(withReplacement = false, fraction, new XORShiftRandom(seed).nextInt())
  } else { //离散型特征
      input.sparkContext.emptyRDD[LabeledPoint]
    }

    // 调用排序及切分函数
    findSplitsBySorting(sampledInput, metadata, continuousFeatures)
  }

  private def findSplitsBySorting(
      input: RDD[LabeledPoint],
      metadata: DecisionTreeMetadata,
      continuousFeatures: IndexedSeq[Int]): Array[Array[Split]] = {

    val continuousSplits: scala.collection.Map[Int, Array[Split]] = {
      // reduce the parallelism for split computations when there are less
      // continuous features than input partitions. this prevents tasks from
      // being spun up that will definitely do no work.
      val numPartitions = math.min(continuousFeatures.length, input.partitions.length)

      input
        .flatMap(point => continuousFeatures.map(idx => (idx, point.features(idx))))
        .groupByKey(numPartitions)
        .map { case (idx, samples) =>
          val thresholds = findSplitsForContinuousFeature(samples, metadata, idx)
          val splits: Array[Split] = thresholds.map(thresh => new ContinuousSplit(idx, thresh))
          logDebug(s"featureIndex = $idx, numSplits = ${splits.length}")
          (idx, splits)
        }.collectAsMap()
    }

    val numFeatures = metadata.numFeatures
    val splits: Array[Array[Split]] = Array.tabulate(numFeatures) {
      // 连续型特征
      case i if metadata.isContinuous(i) =>
        val split = continuousSplits(i)
        metadata.setNumSplits(i, split.length)
        split

      case i if metadata.isCategorical(i) && metadata.isUnordered(i) =>
        // Unordered features
        // 2^(maxFeatureValue - 1) - 1 combinations
        // 离散型无序特征
        val featureArity = metadata.featureArity(i)
        Array.tabulate[Split](metadata.numSplits(i)) { splitIndex =>
          // 提取特征的属性值，返回集合包含其中一个或者多个的离散属性值
          val categories = extractMultiClassCategories(splitIndex + 1, featureArity)
          new CategoricalSplit(i, categories.toArray, featureArity)
        }

      case i if metadata.isCategorical(i) =>
        // Ordered features
        //   Splits are constructed as needed during training.
        // 离散型有序特征，无需计算，训练时创建即可
        Array.empty[Split]
    }
    splits
  }

  /**
   * Nested method to extract list of eligible categories given an index. It extracts the
   * position of ones in a binary representation of the input. If binary
   * representation of an number is 01101 (13), the output list should (3.0, 2.0,
   * 0.0). The maxFeatureValue depict the number of rightmost digits that will be tested for ones.
   */
  private[tree] def extractMultiClassCategories(
      input: Int,
      maxFeatureValue: Int): List[Double] = {
    var categories = List[Double]()
    var j = 0
    var bitShiftedInput = input
    while (j < maxFeatureValue) {
      if (bitShiftedInput % 2 != 0) {
        // updating the list of categories.
        categories = j.toDouble :: categories
      }
      // Right shift by one
      bitShiftedInput = bitShiftedInput >> 1
      j += 1
    }
    categories
  }

  /**
   * 寻找连续特征的分裂点的位置
   * NOTE: Returned number of splits is set based on `featureSamples` and
   *       could be different from the specified `numSplits`.
   *       The `numSplits` attribute in the `DecisionTreeMetadata` class will be set accordingly.
   *
   * @param featureSamples feature values of each sample
   * @param metadata decision tree metadata
   *                 NOTE: `metadata.numbins` will be changed accordingly
   *                       if there are not enough splits to be found
   * @param featureIndex feature index to find splits
   * @return array of splits
   */
  private[tree] def findSplitsForContinuousFeature(
      featureSamples: Iterable[Double],
      metadata: DecisionTreeMetadata,
      featureIndex: Int): Array[Double] = {
    require(metadata.isContinuous(featureIndex),
      "findSplitsForContinuousFeature can only be used to find splits for a continuous feature.")

    val splits = {
      // 分裂数
      val numSplits = metadata.numSplits(featureIndex)

      // get count for each distinct value
      // 统计每个离散属性值及出现的次数 map(value,count)
      val (valueCountMap, numSamples) = featureSamples.foldLeft((Map.empty[Double, Int], 0)) {
        case ((m, cnt), x) =>
          (m + ((x, m.getOrElse(x, 0) + 1)), cnt + 1)
      }
      // sort distinct values
      // 对离散属性值进行排序，升序
      val valueCounts = valueCountMap.toSeq.sortBy(_._1).toArray

      // if possible splits is not enough or just enough, just return all possible splits
      // 如果可能的分裂不够，则返回所有可能的分裂
      val possibleSplits = valueCounts.length
      if (possibleSplits <= numSplits) {
        // 每个离散属性值对应一个分裂点
        valueCounts.map(_._1)
      } else {
        // stride between splits
        // 合并部分离散属性到一个分裂点
        // 每个分裂点之前至少要包含的样本数
        val stride: Double = numSamples.toDouble / (numSplits + 1)
        logDebug("stride = " + stride)

        // iterate `valueCount` to find splits
        // 迭代valueCount，用于查找splits
        val splitsBuilder = mutable.ArrayBuilder.make[Double]
        var index = 1
        // currentCount: sum of counts of values that have been visited
        // 从第一个数组开始扫描，值为特征索引的离散属性值所包含的个数
        var currentCount = valueCounts(0)._2
        // targetCount: target value for `currentCount`.
        // If `currentCount` is closest value to `targetCount`,
        // then current value is a split threshold.
        // After finding a split threshold, `targetCount` is added by stride.
        var targetCount = stride
        while (index < valueCounts.length) {
          val previousCount = currentCount
          currentCount += valueCounts(index)._2
          val previousGap = math.abs(previousCount - targetCount)
          val currentGap = math.abs(currentCount - targetCount)
          // If adding count of current value to currentCount
          // makes the gap between currentCount and targetCount smaller,
          // previous value is a split threshold.
          if (previousGap < currentGap) {
            // 把满足条件的特征索引的值加到split数组中
            splitsBuilder += valueCounts(index - 1)._1
            targetCount += stride
          }
          index += 1
        }

        splitsBuilder.result()
      }
    }

    // TODO: Do not fail; just ignore the useless feature.
    assert(splits.length > 0,
      s"DecisionTree could not handle feature $featureIndex since it had only 1 unique value." +
        "  Please remove this feature and then try again.")

    splits
  }

  private[tree] class NodeIndexInfo(
      val nodeIndexInGroup: Int,
      val featureSubset: Option[Array[Int]]) extends Serializable

  /**
   * Pull nodes off of the queue, and collect a group of nodes to be split on this iteration.
   * This tracks the memory usage for aggregates and stops adding nodes when too much memory
   * will be needed; this allows an adaptive number of nodes since different nodes may require
   * different amounts of memory (if featureSubsetStrategy is not "all").
   *
   * @param nodeQueue  Queue of nodes to split.
   * @param maxMemoryUsage  Bound on size of aggregate statistics.
   * @return  (nodesForGroup, treeToNodeToIndexInfo).
   *          nodesForGroup holds the nodes to split: treeIndex --> nodes in tree.
   *
   *          treeToNodeToIndexInfo holds indices selected features for each node:
   *            treeIndex --> (global) node index --> (node index in group, feature indices).
   *          The (global) node index is the index in the tree; the node index in group is the
   *           index in [0, numNodesInGroup) of the node in this group.
   *          The feature indices are None if not subsampling features.
   */
  private[tree] def selectNodesToSplit(
      nodeQueue: mutable.Queue[(Int, LearningNode)],
      maxMemoryUsage: Long,
      metadata: DecisionTreeMetadata,
      rng: Random): (Map[Int, Array[LearningNode]], Map[Int, Map[Int, NodeIndexInfo]]) = {
    // Collect some nodes to split:
    //  nodesForGroup(treeIndex) = nodes to split
    val mutableNodesForGroup = new mutable.HashMap[Int, mutable.ArrayBuffer[LearningNode]]()
    val mutableTreeToNodeToIndexInfo =
      new mutable.HashMap[Int, mutable.HashMap[Int, NodeIndexInfo]]()
    var memUsage: Long = 0L
    var numNodesInGroup = 0
    // If maxMemoryInMB is set very small, we want to still try to split 1 node,
    // so we allow one iteration if memUsage == 0.
    while (nodeQueue.nonEmpty && (memUsage < maxMemoryUsage || memUsage == 0)) {
      val (treeIndex, node) = nodeQueue.head
      // Choose subset of features for node (if subsampling).
      val featureSubset: Option[Array[Int]] = if (metadata.subsamplingFeatures) {
        Some(SamplingUtils.reservoirSampleAndCount(Range(0,
          metadata.numFeatures).iterator, metadata.numFeaturesPerNode, rng.nextLong())._1)
      } else {
        None
      }
      // Check if enough memory remains to add this node to the group.
      val nodeMemUsage = RandomForest.aggregateSizeForNode(metadata, featureSubset) * 8L
      if (memUsage + nodeMemUsage <= maxMemoryUsage || memUsage == 0) {
        nodeQueue.dequeue()
        mutableNodesForGroup.getOrElseUpdate(treeIndex, new mutable.ArrayBuffer[LearningNode]()) +=
          node
        mutableTreeToNodeToIndexInfo
          .getOrElseUpdate(treeIndex, new mutable.HashMap[Int, NodeIndexInfo]())(node.id)
          = new NodeIndexInfo(numNodesInGroup, featureSubset)
      }
      numNodesInGroup += 1
      memUsage += nodeMemUsage
    }
    if (memUsage > maxMemoryUsage) {
      // If maxMemoryUsage is 0, we should still allow splitting 1 node.
      logWarning(s"Tree learning is using approximately $memUsage bytes per iteration, which" +
        s" exceeds requested limit maxMemoryUsage=$maxMemoryUsage. This allows splitting" +
        s" $numNodesInGroup nodes in this iteration.")
    }
    // Convert mutable maps to immutable ones.
    val nodesForGroup: Map[Int, Array[LearningNode]] =
      mutableNodesForGroup.mapValues(_.toArray).toMap
    val treeToNodeToIndexInfo = mutableTreeToNodeToIndexInfo.mapValues(_.toMap).toMap
    (nodesForGroup, treeToNodeToIndexInfo)
  }

  /**
   * Get the number of values to be stored for this node in the bin aggregates.
   *
   * @param featureSubset  Indices of features which may be split at this node.
   *                       If None, then use all features.
   */
  private def aggregateSizeForNode(
      metadata: DecisionTreeMetadata,
      featureSubset: Option[Array[Int]]): Long = {
    val totalBins = if (featureSubset.nonEmpty) {
      featureSubset.get.map(featureIndex => metadata.numBins(featureIndex).toLong).sum
    } else {
      metadata.numBins.map(_.toLong).sum
    }
    if (metadata.isClassification) {
      metadata.numClasses * totalBins
    } else {
      3 * totalBins
    }
  }

}
