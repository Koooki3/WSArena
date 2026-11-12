# baselines-mw-v1 / a1

| model                   | effort   | ckpt_class   |   runs |   verified |   completion |   valid_patch |   dSR_id |   dSR_ood |   recovery |   regression |   frac_changed |
|:------------------------|:---------|:-------------|-------:|-----------:|-------------:|--------------:|---------:|----------:|-----------:|-------------:|---------------:|
| scripted-analytic_ridge | none     | S1           |     13 |         13 |            1 |          0.25 |    0.638 |     0.385 |      0.998 |        0.003 |          0.06  |
| scripted-analytic_ridge | none     | S2           |      6 |          6 |            1 |          0.25 |    0.328 |     0.43  |      0.653 |        0.164 |          0.06  |
| scripted-analytic_ridge | none     | S3           |      6 |          6 |            1 |          0.25 |    0.023 |    -0.088 |      0.419 |        0.1   |          0.06  |
| scripted-gain_greedy    | none     | S1           |     26 |         26 |            1 |          1    |    0.228 |     0.224 |      0.451 |        0.218 |          0.058 |
| scripted-gain_greedy    | none     | S2           |     12 |         12 |            1 |          1    |    0.102 |     0.095 |      0.328 |        0.33  |          0.058 |
| scripted-gain_greedy    | none     | S3           |     12 |         12 |            1 |          1    |    0.022 |     0.073 |      0.296 |        0.077 |          0.058 |
| scripted-null           | none     | S1           |     13 |         13 |            1 |          0    |    0     |     0     |      0     |        0     |          0     |
| scripted-null           | none     | S2           |      6 |          6 |            1 |          0    |    0     |     0     |      0     |        0     |          0     |
| scripted-null           | none     | S3           |      6 |          6 |            1 |          0    |    0     |     0     |      0     |        0     |          0     |
| scripted-oracle_restore | none     | S1           |     13 |         13 |            1 |          0.25 |    0.635 |     0.502 |      0.993 |        0.003 |          0.007 |
| scripted-oracle_restore | none     | S2           |      6 |          6 |            1 |          0    |    0     |     0     |      0     |        0     |          0     |
| scripted-oracle_restore | none     | S3           |      6 |          6 |            1 |          0    |    0     |     0     |      0     |        0     |          0     |
| scripted-random         | none     | S1           |     26 |         26 |            1 |          1    |   -0.359 |    -0.215 |      0     |        1     |          0.189 |
| scripted-random         | none     | S2           |     12 |         12 |            1 |          1    |   -0.372 |    -0.005 |      0     |        0.996 |          0.189 |
| scripted-random         | none     | S3           |     12 |         12 |            1 |          1    |   -0.758 |    -0.455 |      0     |        1     |          0.189 |
| scripted-random_greedy  | none     | S1           |     26 |         26 |            1 |          1    |   -0.227 |    -0.122 |      0.081 |        0.793 |          0.028 |
| scripted-random_greedy  | none     | S2           |     12 |         12 |            1 |          1    |   -0.318 |    -0.008 |      0.061 |        0.931 |          0.025 |
| scripted-random_greedy  | none     | S3           |     12 |         12 |            1 |          1    |   -0.658 |    -0.34  |      0.057 |        0.88  |          0.022 |
