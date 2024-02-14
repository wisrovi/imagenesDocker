from test_tracker_2 import evaluate
from genetic_algoritm import GAParams, GA

if __name__ == "__main__":
    base_optimize = dict(
        conf=float,
        iou=float,
        retina_masks=bool,
        half=bool,
        track_high_thresh=float,
        track_low_thresh=float,
        new_track_thresh=float,
        track_buffer=int,
        match_thresh=float,
        proximity_thresh=float,
        appearance_thresh=float,
        with_reid=bool,
    )

    ga_params = GAParams(
        population_size=50,
        evolution_generations=100,
        params_to_optimize=base_optimize,
    )

    ga_instance = GA(ga_params, evaluate)

    best_individual = ga_instance.evolve_population()

    print("Best Individual:", best_individual)

    # save best individual using yaml file
    import yaml

    with open("best_individual.yaml", "w") as f:
        yaml.dump(best_individual, f)
