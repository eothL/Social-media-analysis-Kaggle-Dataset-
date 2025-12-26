# Shallow analysis
## Overall numeric summary

|                                                  | mean  |
| ------------------------------------------------ | ----- |
| Average daily usage(hours)                       | 4.919 |
| Sleep per night (hours)                          | 6.867 |
| Mental health score (higher is better, up to 10) | 6.227 |
| Conflicts over social media                      | 2.850 |
| Addicted score (lower is better, up to 10)       | 6.437 |


| type                      | count |
| ------------------------- | ----- |
| female                    | 353   |
| male                      | 352   |
| undergraduate             | 353   |
| graduate                  | 325   |
| high school               | 27    |
| instagram                 | 249   |
| titktok                   | 154   |
| facebook                  | 123   |
| whatsapp                  | 54    |
| twitter                   | 30    |
| linkedin                  | 21    |
| single                    | 384   |
| in relationship           | 289   |
| complicated               | 32    |
| yes(academic performance) | 453   |
| no(academic performance)  | 252   |
most represented country : india, USA, Canada. France, Mexico ...
  
## Age analyse

We can see there is difference between 18 and 24, people tends to less watch their phones but the distribution is less uniform and more spread as people grew up. This can be due either generation differences or people having less time to scroll on social media.
This difference could also be explain by the fewer number of high school student in the pool of data. Overall it may be very similar the consumption of social media 
## Gender analyse
Female tends to spend more time in average in social media than male and standard deviation higher than men and less sleeping hours for women which result in lower score in mental health score for women in average and more conflicts over social media and increase addiction score.

the social media usage is more sparse for women than man. We can see that in average man look at the social media 0.2 hours less and standard deviation inferior and have a better mental health score and sleep hours is better and have less conflicts over social media and feel less addicted


## Academic level vs high school deltas
we compare against high school student : we find  out that in average after highschool people tend to spend less time on social media, sleep more and have better mental health score less addicted score and conflicts over social media

## Country analyses 
USA has the most usage hours of all country in average and much more addicted

## Platform usage
People tends to spend more time on tiktok
more addicted platform has the most usage and lowest mental health

people on whatsapp called people a lot or communicate with people a lot but still has a really low mental health score despite using an app with no addiction tools  because one of the lowest sleeping hours in average and a lot of conflicts over social media. high addiction score, higher than tiktok.

## relationship analyse
being relationship or not doesn't influence that much the daily usage of social media but people struggling with their partners tends to use less social media.
complicated people tends to sleep less than single and being relationship people with 1 hour less in average and they have a lower mental health score than single and being relationship people and 
they have more conflicts over social media and are way more addicted to social media despite having less daily usage hours

## academic performance
more than 64.3% think that their academic performance are impacted by the use of social media.
As we can see, people that said yes has 1.7 more hours in daily usage in average than people who said no and tends to sleep more (+1.6 hours) and has a better mental health score and has less conflicts over social media and are less addicted.

People are trapped into a negative cycle where the more they use social media, they less sleeping time they have and the more addicted they are and the less they are feeling good which tends to strengthen the social media usage.



## other analyses
Pearson and spearman coefficient are almost identical, we can conclude that the relationships among the numeric variables are mostly lienar and monotonic, with no strong outliers or non lienar patterns driving the association.

  - Pearson ≈ Spearman ⇒ rank-order and linear relationships are aligned.                                                                 
  - No major nonlinear or threshold effects are evident in the pairwise associations.                                                     
  - Using Pearson is fine for interpretation and regression modeling, but Spearman is a useful robustness check.


correlation matrix
![[pearson correlation matrix]](image-1.png)



# In depth analyse
## Academic Level Sample Size Check

## Group sizes
- High School: 27
- Undergraduate: 353
- Graduate: 325

## 1) Confidence intervals for group means (95%)
Normal-approximate 95% CIs for each outcome by academic level:

| Outcome | High School (mean, CI) | Undergraduate (mean, CI) | Graduate (mean, CI) |
|---|---|---|---|
| Avg_Daily_Usage_Hours | 5.54 [5.28, 5.80] | 5.00 [4.87, 5.14] | 4.78 [4.64, 4.91] |
| Addicted_Score | 8.04 [7.61, 8.46] | 6.49 [6.32, 6.66] | 6.24 [6.08, 6.40] |
| Mental_Health_Score | 5.11 [4.89, 5.33] | 6.18 [6.06, 6.30] | 6.37 [6.26, 6.49] |
| Sleep_Hours_Per_Night | 5.46 [5.30, 5.61] | 6.83 [6.70, 6.95] | 7.03 [6.92, 7.14] |
| Conflicts_Over_Social_Media | 3.74 [3.43, 4.05] | 2.92 [2.82, 3.01] | 2.70 [2.60, 2.81] |

Interpretation: High School intervals are wider due to n=27; treat HS comparisons as exploratory.

## 2) Sensitivity check: correlations with and without High School
If correlations are stable after excluding High School, results are less sensitive to the small group.

| Pair | All students (r) | Excluding High School (r) | Δ |
|---|---|---|---|
| Avg_Daily_Usage_Hours vs Mental_Health_Score | -0.801 | -0.801 | +0.000 |
| Avg_Daily_Usage_Hours vs Sleep_Hours_Per_Night | -0.791 | -0.794 | -0.004 |
| Avg_Daily_Usage_Hours vs Conflicts_Over_Social_Media | 0.805 | 0.804 | -0.000 |
| Addicted_Score vs Mental_Health_Score | -0.945 | -0.944 | +0.001 |
| Addicted_Score vs Sleep_Hours_Per_Night | -0.765 | -0.753 | +0.012 |
| Addicted_Score vs Conflicts_Over_Social_Media | 0.934 | 0.932 | -0.001 |

Interpretation: Small deltas indicate that excluding HS does not materially change the main relationships.

## 3) Effect sizes (Cohen’s d) for High School vs other levels
Magnitude guide: ~0.2 small, ~0.5 medium, ~0.8 large (direction shows HS relative to comparison).

| Outcome | HS vs Undergraduate (d) | HS vs Graduate (d) |
|---|---|---|
| Avg_Daily_Usage_Hours | 0.43 | 0.64 |
| Addicted_Score | 0.96 | 1.22 |
| Mental_Health_Score | -0.98 | -1.22 |
| Sleep_Hours_Per_Night | -1.19 | -1.63 |
| Conflicts_Over_Social_Media | 0.90 | 1.09 |

Interpretation: Use effect sizes rather than p-values given the small HS sample.

