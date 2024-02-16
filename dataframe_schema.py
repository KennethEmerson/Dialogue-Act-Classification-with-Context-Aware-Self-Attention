#######################################################################################################
# CASA--DIALOGUE-ACT-CLASSIFIER
# File contains Pandera DataSchema required for all Pandas dataframes used to hold training, validation
# and test data.
#######################################################################################################
import pandera as pa
from pandera.typing import Series

class DataSchema(pa.DataFrameModel):
    dialogue_id: Series[str]
    intent: Series[str] 
    utterance: Series[str]