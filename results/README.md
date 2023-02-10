# Results from Industry 4.0 model case

There are four CSV files available:
* `industry-base.csv` -- baseline results for industry model
* `industry-custom.csv` -- results of our custom networks for industry model
* `synth-base.csv` -- baseline results for synthetic tests
* `synth-custom.csv` -- results of our custom networks for synthetic tests

The `.r` scripts are just for an illustration, how the graphs in the paper were plotted.

All CSV have the same format. First three values are `iteration` (number 1 to 5), `dataset`, and `model` respectively. They are followed by 100 float values -- an accuracy measured after every epoch.

The `dataset` is either `random` or `combined` for the industry example (the terms are explained in the paper) and `c5` or `c10` for synthetic tests. The `cX` name stands for *conjunction* of `X` arbitrary input values.

Models `128` to `1024` refers to single-layer dense networks and `128-128` to `1024-1024` are networks with two hidden layers. The numbers denote the numbers of units in each layer. The `T_ab` and `T_right` models refer to the first two types of relaxation where only time constraints are trained by a model (with above & below and the right time predicates). The `all` model refers to relaxation of all three inputs.

**The synthetic results summary:**

```
c5
         128:  mean = 0.9929161190986633  std.dev = 0.000182315140267509
         256:  mean = 0.9943774342536926  std.dev = 0.00021167422253216497
         512:  mean = 0.9956000089645386  std.dev = 0.0000632111183348894
        1024:  mean = 0.9962935447692871  std.dev = 0.000034438364712977285
        2048:  mean = 0.9964451670646668  std.dev = 0.00004848843347678796
     128-128:  mean = 0.9996387124061584  std.dev = 0.00008927590479468071
     256-256:  mean = 0.9995838761329651  std.dev = 0.00007991715234531822
     512-512:  mean = 0.9994225859642029  std.dev = 0.00011914579742109553
   1024-1024:  mean = 0.9994064569473267  std.dev = 0.00008559589782787358
   2048-2048:  mean = 0.9991645097732544  std.dev = 0.00014477235370372626
       synth:  mean = 0.9980612874031067  std.dev = 0.0006152323193793062
c10
         128:  mean = 0.9988866090774536  std.dev = 0.000022161824312046373
         256:  mean = 0.9990048885345459  std.dev = 0.000021770318571812732
         512:  mean = 0.9990791797637939  std.dev = 0.000011399704836832905
        1024:  mean = 0.9991388082504272  std.dev = 0.000009968748763605958
        2048:  mean = 0.9991700887680054  std.dev = 0.000011315565448974227
     128-128:  mean = 0.9997057676315307  std.dev = 0.000014299828513760397
     256-256:  mean = 0.9996461391448974  std.dev = 0.000038185741086285384
     512-512:  mean = 0.999576735496521   std.dev = 0.00002484506672064777
   1024-1024:  mean = 0.9995200395584106  std.dev = 0.00007278394048572356
   2048-2048:  mean = 0.9994907140731811  std.dev = 0.000032626203877682724
       synth:  mean = 0.998328447341919   std.dev = 0.00014521956056147742
```
The model names for baseline are the same as for industry example. The `synth` model denotes our approach.


# Results from ReCodEx code assignments model case

All results are aggregated in `recodex.csv` file. We have tried networks with one hidden layer ranging from 64 to 256, as the results indicate all models are quite similar in the terms of accuracy. Summarized accuracy values are below:

```
  dense-64:  mean = 0.8600960850715638  std.dev = 0.005464712119836069
 dense-128:  mean = 0.8552030682563782  std.dev = 0.004716171020515704
 dense-256:  mean = 0.8566048502922058  std.dev = 0.0043046716388230365
 custom-64:  mean = 0.8485629439353943  std.dev = 0.008478090237982301
custom-128:  mean = 0.8551486968994141  std.dev = 0.007964714993054837
custom-256:  mean = 0.8497487545013428  std.dev = 0.00680697627487396
```

The `dense-` models are baseline, `custom-` models employ our modification that aims to test whether fuzzified rules can be integrated into classification problems and combined with other parts of the network. Both approaches show very similar results in the terms of accuracy, hence our modification does not hinder the learning capabilities of the network.

