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
        if configs.model_gen_query == "CatBoostRanker":
            logging.info(f"RankerSolution")
            configs.data_manager_.configure_label_index_mapping()
            return ScikitSolution.CatBoostRankerSolution.from_configs(
                args=args, configs=configs, output_data_path=output_data_path)

        if configs.model_gen_query.endswith("Ranker"):
            logging.info(f"RankerSolution")
            configs.data_manager_.configure_label_index_mapping()
            return ScikitSolution.RankerSolution.from_configs(
                args=args, configs=configs, output_data_path=output_data_path)

        if configs.model_gen_query.endswith("Classifier"):
            logging.info(f"ClassifierSolution")
            configs.data_manager_.configure_label_index_mapping()
            return ScikitSolution.ClassifierSolution.from_configs(
                args=args, configs=configs, output_data_path=output_data_path)

        return ScikitSolution.RegressorSolution.from_configs(
            args=args, configs=configs, output_data_path=output_data_path)


if "__main__" == __name__:
    pass
