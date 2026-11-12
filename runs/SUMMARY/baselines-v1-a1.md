# baselines-v1 / a1

| model                   | effort   | ckpt_class   |   runs |   verified |   completion |   valid_patch |   dSR_id |   dSR_ood |   recovery |   regression |   frac_changed |
|:------------------------|:---------|:-------------|-------:|-----------:|-------------:|--------------:|---------:|----------:|-----------:|-------------:|---------------:|
| scripted-analytic_ridge | none     | S1           |     28 |         28 |            1 |          0.25 |    0.489 |     0.394 |      0.918 |        0.088 |          0.061 |
| scripted-analytic_ridge | none     | S2           |      9 |          9 |            1 |          0.25 |   -0.016 |     0.157 |      0.373 |        0.497 |          0.063 |
| scripted-analytic_ridge | none     | S3           |     12 |         12 |            1 |          0.25 |    0.175 |     0.49  |      0.78  |        0.02  |          0.062 |
| scripted-null           | none     | S1           |     28 |         28 |            1 |          0    |    0     |     0     |      0     |        0     |          0     |
| scripted-null           | none     | S2           |      9 |          9 |            1 |          0    |    0     |     0     |      0     |        0     |          0     |
| scripted-null           | none     | S3           |     12 |         12 |            1 |          0    |    0     |     0     |      0     |        0     |          0     |
| scripted-oracle_restore | none     | S1           |     28 |         28 |            1 |          0.25 |    0.5   |     0.394 |      0.929 |        0.075 |          0.015 |
| scripted-oracle_restore | none     | S2           |      9 |          9 |            1 |          0    |    0     |     0     |      0     |        0     |          0     |
| scripted-oracle_restore | none     | S3           |     12 |         12 |            1 |          0    |    0     |     0     |      0     |        0     |          0     |
| scripted-random         | none     | S1           |     84 |         84 |            1 |          1    |   -0.2   |    -0.144 |      0.141 |        0.668 |          0.185 |
| scripted-random         | none     | S2           |     27 |         27 |            1 |          1    |   -0.213 |    -0.072 |      0.044 |        0.582 |          0.185 |
| scripted-random         | none     | S3           |     36 |         36 |            1 |          1    |   -0.348 |    -0.124 |      0.174 |        0.526 |          0.185 |
