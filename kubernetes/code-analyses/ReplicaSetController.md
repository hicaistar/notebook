# ReplicaSetController
> A good sample controller for pods, learn it.(kubernetes/pkg/controller/replicaset)

### 基本思路
监听 replicasets 和 pods，有 update 入队列，然后 syncReplicSet

### functions
- deletePod (参考实现)
  - 获取 controllerRef，做一些判断处理
- syncReplicaSet
  - 通过 cache.SplitMetaNamespaceKey 获取 namespace, name
  - 通过 Lister 获取 replicaset
  - 获取 Labelselector
    - metav1.LabelSelectorAsSelector
  - 获取所有的 pods
  - claimPods(selector)
    - 挑选出该 replicaset 的 pods
  - (if needSync && DeletionTimestamp == nil)
    - manageReplicas
  - calculateStatus
  - updateReplicaSetStatus
- manageReplicas
  - diff 出副本数量差别
  - slowStartBatch 批量创建 pod
    - 填充 ownerRefernece 信息
