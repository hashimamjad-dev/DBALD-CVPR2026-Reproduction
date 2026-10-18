# Published Table 1 (reference only)

All values are percentages. B = benign clean ACC; P = poisoned-model clean ACC. ASR columns are published attack results, never measurements from this project.

## CULane

|Model|Method|B ACC|P ACC|LDA ASR|LRA ASR|LOA ASR|Avg ASR|
|---|---|---|---|---|---|---|---|
|LaneATT|BadNets|75.01|72.79|49.27|43.37|47.36|46.67|
|LaneATT|Blended|75.01|73.15|57.14|59.07|47.04|54.41|
|LaneATT|LD-Attack|75.01|72.48|71.24|56.24|63.71|63.73|
|LaneATT|BadLANE|75.01|73.78|64.33|45.22|54.15|54.56|
|LaneATT|DBALD|75.01|74.20|81.65|73.45|74.19|76.41|
|ADNet|BadNets|85.51|79.60|50.13|54.33|61.39|55.28|
|ADNet|Blended|85.51|74.37|43.67|56.42|56.51|52.20|
|ADNet|LD-Attack|85.51|79.14|59.47|63.39|67.15|63.34|
|ADNet|BadLANE|85.51|85.46|36.61|61.16|65.95|54.57|
|ADNet|DBALD|85.51|82.37|66.49|64.82|68.17|66.50|
|SCNN|BadNets|69.75|69.52|74.39|63.63|62.76|66.93|
|SCNN|Blended|69.75|63.98|75.14|63.45|61.60|66.73|
|SCNN|LD-Attack|69.75|64.38|74.83|62.21|64.17|67.07|
|SCNN|BadLane|69.75|67.57|72.28|63.96|62.94|66.39|
|SCNN|DBALD|69.75|68.75|76.19|67.34|68.69|70.74|
|RESA|BadNets|78.98|76.68|71.08|58.68|69.14|66.30|
|RESA|Blended|78.98|75.87|73.85|59.64|57.76|63.75|
|RESA|LD-Attack|78.98|71.35|73.26|64.77|69.18|69.07|
|RESA|BadLANE|78.98|72.81|75.39|62.94|68.23|68.85|
|RESA|DBALD|78.98|77.39|78.15|65.15|76.86|73.39|

## TuSimple

|Model|Method|B ACC|P ACC|LDA ASR|LOA ASR|Avg ASR|
|---|---|---|---|---|---|---|
|LaneATT|BadNets|96.81|95.53|45.62|75.63|60.63|
|LaneATT|Blended|96.81|96.27|46.31|72.47|59.39|
|LaneATT|LD-Attack|96.81|96.21|88.17|90.27|89.22|
|LaneATT|BadLANE|96.81|96.18|89.25|89.76|89.51|
|LaneATT|DBALD|96.81|95.72|90.04|91.63|90.84|
|ADNet|BadNets|96.87|95.46|50.34|52.98|51.66|
|ADNet|Blended|96.87|95.10|56.27|59.12|57.70|
|ADNet|LD-Attack|96.87|96.35|53.24|66.32|59.78|
|ADNet|BadLANE|96.87|96.29|59.17|67.48|63.33|
|ADNet|DBALD|96.87|96.61|89.24|72.07|80.66|
|SCNN|BadNets|93.78|91.56|64.72|62.79|63.76|
|SCNN|Blended|93.78|91.37|58.30|64.13|61.22|
|SCNN|LD-Attack|93.78|92.23|94.08|91.43|92.76|
|SCNN|BadLane|93.78|90.89|92.19|88.57|90.38|
|SCNN|DBALD|93.78|92.28|95.78|91.29|93.54|
|RESA|BadNets|96.81|96.63|64.78|73.46|69.12|
|RESA|Blended|96.81|96.65|68.11|57.28|62.70|
|RESA|LD-Attack|96.81|96.67|92.53|92.04|92.29|
|RESA|BadLANE|96.81|96.42|92.47|91.19|91.83|
|RESA|DBALD|96.81|96.67|96.57|94.28|95.43|

The CSV is integrity-checked against its manifest. Do not recompute published averages or overwrite it with experimental results. See paper_analysis.md for inconsistencies.