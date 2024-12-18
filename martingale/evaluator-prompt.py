from martingale.stats.fewvar import FEWVar



def evaluator(gen, model_cls_list):
    """

        A generator that yields updated estimates of relative performance
        when we feed a process like brown_gen to a list of nowcasters.

        Similar in spirit to this plotting code but instead of plotting we maintain running estimates of
        the mean loss incurred when obs['x'] is compared to the nowcasts.

        We also use a burn-in period like below.

        But we use multi-processing to compute multiple series at once and run multiple models against them.

        The ergodic average std in dx changes is used to normalize all measures so that the errors in numerically large time-series
        do not dominate the results.



          # Number of total steps and burn-in period
    n = 200
    burn_in = 10000 - n

    # Advance the generator to settle to ergodic average (burn-in)
    count = 0
    for _ in gen:
        count += 1
        if count >= burn_in:
            break

    # Initialize the model
    model = cls()

    # Storage for values
    x_values = []
    y_values = []
    model_means = []
    model_vars = []

    # Now run the model on the next n steps
    for obs in gen:
        # Update the model with the new observation
        model.update(x=obs['x'])
        model_mean = model.get_mean()
        model_var = model.get_var()

        # Append current results
        x_values.append(obs['x'])
        y_values.append(obs['y'])
        model_means.append(model_mean)
        model_vars.append(model_var)


    """