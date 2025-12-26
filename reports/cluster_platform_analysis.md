# Cluster Behavior on Instagram vs TikTok

## Method
- Use the same 6 numeric features as the clustering analysis.
- Assign each student to the nearest KMeans center from the clustering analysis (k=4).
- Filter to Most_Used_Platform in {Instagram, TikTok}.

## Cluster labels (verified)
- Cluster 0 - Balanced Users: low usage (3.84), high sleep (8.08), good mental health (7.39), low addiction (4.70).
- Cluster 1 - High-Risk Addicted Users: highest usage (6.41) and addiction (8.25), lowest sleep (5.61) and mental health (4.99).
- Cluster 2 - Low-Usage Mature Users: oldest (22.12), low usage (3.80), lowest conflicts (1.73), low addiction (4.70).
- Cluster 3 - Moderate-Risk Users: mid usage (5.00) and addiction (6.98), moderate sleep (6.77) and mental health (5.88).

Note: In the clustering notebook mapping, Cluster 0 and Cluster 2 labels appear swapped relative to age; this report uses the label that matches the observed behavior.

## Cluster mix by platform
| Platform | Cluster (label) | Count | Share % |
| --- | --- | --- | --- |
| Instagram | Cluster 0 (Balanced Users) | 70 | 28.1 |
| Instagram | Cluster 1 (High-Risk Addicted Users) | 59 | 23.7 |
| Instagram | Cluster 2 (Low-Usage Mature Users) | 18 | 7.2 |
| Instagram | Cluster 3 (Moderate-Risk Users) | 102 | 41.0 |
| TikTok | Cluster 0 (Balanced Users) | 13 | 8.4 |
| TikTok | Cluster 1 (High-Risk Addicted Users) | 68 | 44.2 |
| TikTok | Cluster 2 (Low-Usage Mature Users) | 5 | 3.2 |
| TikTok | Cluster 3 (Moderate-Risk Users) | 68 | 44.2 |

## Within-cluster platform averages
| Cluster (label) | Platform | N | Avg_Daily_Usage_Hours | Sleep_Hours_Per_Night | Mental_Health_Score | Conflicts_Over_Social_Media | Addicted_Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cluster 0 (Balanced Users) | Instagram | 70 | 3.70 | 8.22 | 7.44 | 2.13 | 4.79 |
| Cluster 0 (Balanced Users) | TikTok | 13 | 3.96 | 7.85 | 7.08 | 2.38 | 5.38 |
| Cluster 1 (High-Risk Addicted Users) | Instagram | 59 | 6.56 | 5.62 | 4.75 | 4.10 | 8.49 |
| Cluster 1 (High-Risk Addicted Users) | TikTok | 68 | 6.15 | 5.45 | 5.15 | 4.06 | 8.29 |
| Cluster 2 (Low-Usage Mature Users) | Instagram | 18 | 4.03 | 7.68 | 7.44 | 1.72 | 4.72 |
| Cluster 2 (Low-Usage Mature Users) | TikTok | 5 | 4.64 | 7.46 | 7.00 | 2.40 | 5.40 |
| Cluster 3 (Moderate-Risk Users) | Instagram | 102 | 4.85 | 6.89 | 5.78 | 3.04 | 6.97 |
| Cluster 3 (Moderate-Risk Users) | TikTok | 68 | 4.86 | 6.91 | 5.93 | 3.03 | 7.10 |

## Key takeaways
- TikTok is concentrated in higher-risk segments: 88.3% of TikTok users fall into Moderate-Risk or High-Risk clusters, versus 64.7% on Instagram.
- The largest platform gap shows up inside the low-risk clusters: TikTok users in Cluster 0 and Cluster 2 show higher addiction and slightly worse sleep/mental health than Instagram peers.
- Once users are in the High-Risk cluster, platform differences are small; addiction stays around 8.3 to 8.5 regardless of platform.
- The Moderate-Risk cluster is nearly identical across platforms, suggesting the overall Instagram vs TikTok gap is driven more by cluster mix than by within-cluster behavior.

Note: Cluster 2 on TikTok has a small sample (n=5), so its platform gap should be treated as directional rather than definitive.
