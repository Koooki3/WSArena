# baselines-mwpeg-v1 / a1

| model                   | effort   | ckpt_class   |   runs |   verified |   completion |   valid_patch |   dSR_id |   dSR_ood |   recovery |   regression |   frac_changed |
|:------------------------|:---------|:-------------|-------:|-----------:|-------------:|--------------:|---------:|----------:|-----------:|-------------:|---------------:|
| scripted-analytic_ridge | none     | S1           |      6 |          6 |            1 |          0.25 |    0.342 |     0.2   |      0.689 |        0.143 |          0.06  |
| scripted-analytic_ridge | none     | S2           |      2 |          2 |            1 |          0.25 |    0.355 |     0.725 |      0.771 |        0.434 |          0.06  |
| scripted-analytic_ridge | none     | S3           |      3 |          3 |            1 |          0.25 |    0.087 |     0.007 |      0.398 |        0.027 |          0.06  |
| scripted-gain_greedy    | none     | S1           |     12 |         12 |            1 |          1    |    0.01  |     0.093 |      0.228 |        0.254 |          0.058 |
| scripted-gain_greedy    | none     | S2           |      4 |          4 |            1 |          1    |    0.195 |     0.415 |      0.465 |        0.32  |          0.058 |
| scripted-gain_greedy    | none     | S3           |      6 |          6 |            1 |          1    |   -0.06  |    -0.017 |      0.286 |        0.193 |          0.058 |
| scripted-null           | none     | S1           |      6 |          6 |            1 |          0    |    0     |     0     |      0     |        0     |          0     |
| scripted-null           | none     | S2           |      2 |          2 |            1 |          0    |    0     |     0     |      0     |        0     |          0     |
| scripted-null           | none     | S3           |      3 |          3 |            1 |          0    |    0     |     0     |      0     |        0     |          0     |
| scripted-oracle_restore | none     | S1           |      6 |          6 |            1 |          0.25 |    0.415 |     0.16  |      0.763 |        0.068 |          0.009 |
| scripted-oracle_restore | none     | S2           |      2 |          2 |            1 |          0    |    0     |     0     |      0     |        0     |          0     |
| scripted-oracle_restore | none     | S3           |      3 |          3 |            1 |          0    |    0     |     0     |      0     |        0     |          0     |
| scripted-random         | none     | S1           |     12 |         12 |            1 |          1    |   -0.415 |    -0.11  |      0     |        1     |          0.189 |
| scripted-random         | none     | S2           |      4 |          4 |            1 |          1    |   -0.345 |    -0.115 |      0     |        1     |          0.189 |
| scripted-random         | none     | S3           |      6 |          6 |            1 |          1    |   -0.72  |    -0.38  |      0     |        1     |          0.189 |
| scripted-random_greedy  | none     | S1           |     12 |         12 |            1 |          1    |   -0.112 |     0.026 |      0.258 |        0.624 |          0.027 |
| scripted-random_greedy  | none     | S2           |      4 |          4 |            1 |          1    |   -0.102 |    -0.02  |      0.141 |        0.564 |          0.022 |
| scripted-random_greedy  | none     | S3           |      6 |          6 |            1 |          1    |   -0.282 |     0.02  |      0.335 |        0.523 |          0.029 |
