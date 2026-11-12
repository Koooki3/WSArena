# baselines-v0 / a2

| model                   | effort   | ckpt_class   |   runs |   verified |   completion |   valid_patch |   dSR_id |   dSR_ood |   recovery |   regression |   frac_changed |
|:------------------------|:---------|:-------------|-------:|-----------:|-------------:|--------------:|---------:|----------:|-----------:|-------------:|---------------:|
| scripted-analytic_ridge | none     | S1           |     28 |         28 |            1 |          0.25 |    0.489 |     0.214 |      0.918 |        0.088 |          0.061 |
| scripted-analytic_ridge | none     | S2           |      9 |          9 |            1 |          0.25 |   -0.016 |     0.068 |      0.373 |        0.497 |          0.063 |
| scripted-analytic_ridge | none     | S3           |     10 |         10 |            1 |          0.25 |    0.136 |     0.122 |      0.886 |        0.106 |          0.061 |
| scripted-null           | none     | S1           |     28 |         28 |            1 |          0    |    0     |     0     |      0     |        0     |          0     |
| scripted-null           | none     | S2           |      9 |          9 |            1 |          0    |    0     |     0     |      0     |        0     |          0     |
| scripted-null           | none     | S3           |     10 |         10 |            1 |          0    |    0     |     0     |      0     |        0     |          0     |
| scripted-oracle_restore | none     | S1           |     28 |         28 |            1 |          0.25 |    0.5   |     0.209 |      0.929 |        0.075 |          0.015 |
| scripted-oracle_restore | none     | S2           |      9 |          9 |            1 |          0    |    0     |     0     |      0     |        0     |          0     |
| scripted-oracle_restore | none     | S3           |     10 |         10 |            1 |          0    |    0     |     0     |      0     |        0     |          0     |
| scripted-random         | none     | S1           |     84 |         84 |            1 |          1    |   -0.2   |    -0.073 |      0.141 |        0.668 |          0.185 |
| scripted-random         | none     | S2           |     27 |         27 |            1 |          1    |   -0.213 |     0     |      0.044 |        0.582 |          0.185 |
| scripted-random         | none     | S3           |     30 |         30 |            1 |          1    |   -0.437 |    -0.137 |      0.154 |        0.641 |          0.185 |
