import efplt
import argparse


def parse():
    parser = argparse.ArgumentParser(description='Experiment Runner')

    def add_options_to_group(options, group):
        for c in options:
            group.add_argument(c[0], action="store_true", help=c[1])

    experiment_choices = [
        ["-misspec", "Misspecification experiment"],
        ["-optim", "Optimization experiment"],
        ["-vecfield", "Vector field visualization"],
        ["-vecfield_quantized", "Quantized Vector field visualization"],
        ["-logistic_regression", "Quantized Logistic Regression visualization"],
    ]

    add_options_to_group(experiment_choices, parser.add_argument_group('Experiment selection').add_mutually_exclusive_group(required=True))

    main_options = [
        ["-run", "Runs the experiment and save results as a .pk file"],
        ["-plot", "Plots the result from a .pk file (requires -save and/or -show)"],
        ["-appendix", "Also run/plot the experiments in the appendix"],
    ]
    add_options_to_group(main_options, parser.add_argument_group('Action selection', "At least one of [-run, -plot] is required"))

    plotting_options = [
        ["-save", "Save the plots"],
        ["-show", "Show the plots"],
    ]
    add_options_to_group(plotting_options, parser.add_argument_group('Plotting options', "At least one of [-save, -show] is required if plotting"))

    args = parser.parse_args()

    if not (args.run or args.plot):
        parser.error('No action requested, add -run and/or -plot')
    if args.plot and not (args.show or args.save):
        parser.error("-plot requires -save and/or -show.")

    return args


def savefigs(figs, expname):
    for i, fig in enumerate(figs):
        efplt.save(fig, expname + "-" + str(i) + ".pdf")


if __name__ == "__main__":
    args = parse()
    print(args)
    # exit()
    print("")

    if args.vecfield:
        import vecfield.main as exp
        expname = "vecfield"
    elif args.vecfield_quantized:
        import vecfield_quantized.quantizer as quantizer
        # q = quantizer.Quantizer(num_bits=32, q_min=-1, q_max=5)
        # q = quantizer.Quantizer(num_bits=2, q_min=-1.1659, q_max=4.375)
        q = quantizer.Quantizer(num_bits=4, q_min=-.5, q_max=4.5)
        # q = quantizer.Quantizer(num_bits=4, q_min=-1, q_max=4.5)

        # q = quantizer.Quantizer(num_bits=4, q_min=-.66, q_max=3.667)
        # q = quantizer.Quantizer(num_bits=2, q_min=-.66, q_max=3.667)
        import vecfield_quantized.main as exp
        expname = "vecfield_quantized"
    elif args.logistic_regression:
        import logistic_regression.quantizer as quantizer
        # q = quantizer.Quantizer(num_bits=3, q_min=-3, q_max=3)
        q = quantizer.Quantizer(num_bits=3, q_min=-3, q_max=2.85)
        import logistic_regression.main as exp
        expname = "logistic_regression"

    if args.misspec:
        import misspec.main as exp
        expname = "misspec"
    if args.optim:
        import optim.main as exp
        expname = "optim"

    if args.run:
        if args.appendix:
            exp.run_appendix()
        else:
            if args.vecfield_quantized or args.logistic_regression:
                exp.run(q)
            else:
                exp.run()

    if args.plot:
        if args.appendix:
            figs = exp.plot_appendix()
        else:
            if args.vecfield_quantized or args.logistic_regression:
                figs = exp.plot(q)
            else:
                figs = exp.plot()

        if args.show:
            efplt.plt.show()

        if args.save:
            savefigs(figs, expname + ("-apx" if args.appendix else ""))
