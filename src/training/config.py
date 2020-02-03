class Config:

    @staticmethod
    def get_model_name() -> str:
        return "CnnBilstm"

    @staticmethod
    def get_batch_size() -> int:
        return 16

    @staticmethod
    def get_epochs() -> int:
        return 200

    @staticmethod
    def get_dataset_path() -> str:
        return "../../data/All_intent.csv"
