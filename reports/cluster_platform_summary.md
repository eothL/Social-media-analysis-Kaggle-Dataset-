# Summary of Cluster Verification and TikTok vs Instagram Analysis

## What was verified
- The 4 clusters from the clustering notebook match the expected behavioral patterns based on their center values.
- Two labels appear swapped in the notebook mapping:
  - Cluster 2 is the oldest and lowest-conflict group and fits **Low-Usage Mature Users**.
  - Cluster 0 fits **Balanced Users** (low usage, high sleep, good mental health).
- The other two labels align:
  - Cluster 1 = **High-Risk Addicted Users**.
  - Cluster 3 = **Moderate-Risk Users**.

## What was added
- A new analysis file was created in `reports/cluster_platform_analysis.md`.
- It includes:
  - Cluster mix by platform (Instagram vs TikTok).
  - Within-cluster averages for usage, sleep, mental health, conflicts, and addiction.
  - Key takeaways about how platform differences are driven by cluster composition.

## Key findings (high level)
- TikTok users are heavily concentrated in Moderate-Risk and High-Risk clusters.
- Platform differences are small inside the High-Risk and Moderate-Risk clusters.
- The biggest platform gaps appear inside low-risk clusters, but the TikTok sample there is small.
