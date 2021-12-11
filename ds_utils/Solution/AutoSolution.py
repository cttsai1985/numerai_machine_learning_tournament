import logging
from argparse import Namespace
from ds_utils.Solution.BaseSolution import ISolution
from ds_utils.Solution import ScikitSolution


class AutoSolution(ISolution):
    def evaluate(self, train_data_type: str = "training", valid_data_type: str = "validation", ):
        raise NotImplementedError()

    def run(self, train_data_type: str = "training", infer_data_type: str = "tournament", ):
        raise NotImplementedError()

    @classmethod
    def from_configs(cls, args: Namespace, configs: "SolutionConfigs", output_data_path: str, **kwargs):
        if configs.model_gen_query.endswith("Regressor"):
            return ScikitSolution.RegressorSolution.from_configs(
                args=args, configs=configs, output_data_path=output_data_path)

        elif configs.model_gen_query.endswith("Classifier"):
            logging.info(f"ClassifierSolution")
            configs.data_manager_.configure_label_index_mapping()
            return ScikitSolution.ClassifierSolution.from_configs(
                args=args, configs=configs, output_data_path=output_data_path)

        if not configs.model_gen_query.endswith("Ranker"):
            raise ValueError(f"No suitable solution found: {configs.model_gen_query}")

        elif configs.model_gen_query == "LGBMRanker":
            logging.info(f"LGBMRankerSolution")
            configs.data_manager_.configure_label_index_mapping()
            return ScikitSolution.LGBMRankerSolution.from_configs(
                args=args, configs=configs, output_data_path=output_data_path)

        elif configs.model_gen_query == "CatBoostRanker":
            logging.info(f"CatBoostRankerSolution")
            return ScikitSolution.CatBoostRankerSolution.from_configs(
                args=args, configs=configs, output_data_path=output_data_path)

        elif configs.model_gen_query == "XGBRanker":
            logging.info(f"XGBRankerSolution")
            return ScikitSolution.XGBRankerSolution.from_configs(
                args=args, configs=configs, output_data_path=output_data_path)

        else:
            logging.info(f"GeneralRankerSolution")
            return ScikitSolution.RankerSolution.from_configs(
                args=args, configs=configs, output_data_path=output_data_path)

        raise ValueError(f"No suitable solution found: {configs.model_gen_query}")


if "__main__" == __name__:
    pass
