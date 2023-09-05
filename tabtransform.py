import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from pytorch_tabular import TabularModel
from pytorch_lightning import Trainer
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
    ExperimentConfig,
)

url = "dataset.csv"
x = pd.read_csv(url)
x.drop("Unnamed: 0", axis=1, inplace=True)
x.head()

embed_columns= [
    'CO_REGIONA', 'CO_MUN_NOT', 'CO_UNI_NOT', 'CS_SEXO', 'CS_RACA',
       'CO_RG_RESI', 'CO_MUN_RES', 'CS_ZONA', 'NOSOCOMIAL', 'FEBRE', 'TOSSE',
       'GARGANTA', 'DISPNEIA', 'DESC_RESP', 'SATURACAO', 'DIARREIA', 'VOMITO',
       'OUTRO_SIN', 'FATOR_RISC', 'ANTIVIRAL', 'HOSPITAL', 'CO_RG_INTE',
       'CO_MU_INTE', 'UTI', 'SUPORT_VEN', 'RAIOX_RES', 'AMOSTRA', 'TP_AMOSTRA',
       'PCR_RESUL', 'CRITERIO', 'DOR_ABD', 'FADIGA', 'PERD_OLFT', 'PERD_PALA',
       'RES_AN', 'ESTRANG', 'VACINA_COV'
]
cont_cols = ["SEM_NOT", "SEM_PRI", "NU_IDADE_N", "CS_GESTANT", "DT_SIN_PRI",
             "DT_INTERNA", "DT_COLETA", "DT_PCR", "DT_EVOLUCA", "TEMP_T", "DOSE_1_COV", "DOSE_2_COV"
]

data_config = DataConfig(
    target=[
        "EVOLUCAO"
    ],  # target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
    continuous_cols=cont_cols,
    categorical_cols=embed_columns,
)

trainer_config = TrainerConfig(
    auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
    batch_size=1024,
    max_epochs=100,
    min_epochs=100,
    accelerator = 'auto',
    early_stopping_patience = 3,

)
optimizer_config = OptimizerConfig()

from pytorch_tabular.models.tab_transformer.config import TabTransformerConfig
model_config = TabTransformerConfig(
    task="classification",
    learning_rate=1e-1,
    #input_embed_dim=32,
    embedding_initialization= "kaiming_uniform",
    #oss="CrossEntropyLoss",
    #share_embedding = True,
    #shared_embedding_fraction=0.25,
    #transformer_activation= "ReLU",
    #embedding_dropout= 0.1,
    metrics = ["accuracy", "f1_score"]
)

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,

)

from sklearn.model_selection import train_test_split
train, test = train_test_split(x, test_size=0.2, random_state=42)
train, val = train_test_split(train, test_size=0.25, random_state=42)

tabular_model.fit(train=train, validation=val)
result = tabular_model.evaluate(test)
pred_df = tabular_model.predict(test)
#tabular_model.save_model("examples/basic")
#loaded_model = TabularModel.load_from_checkpoint("examples/basic")
