import logging
from typing import List, Union

import omegaconf
import pandas as pd
import pymongo
from bson import ObjectId
from omegaconf import OmegaConf
from tqdm import tqdm

from data_lake.constants import DB_ADDRESS, TARGET_DB, DataLakeKey
from shared_lib.constants import DataLakeKeyDict, DatasetInfoKey
from shared_lib.enums import RunMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_OVERWRITE_ALLOWED = {DataLakeKeyDict.HFILE_PATH}


class DatasetHandler:
    def __init__(self) -> None:
        # dataset/collection-specific info
        self.h5_path_key = DataLakeKeyDict.HFILE_PATH
        self.hfile_image_key = DataLakeKeyDict.HFILE_IMAGE
        self.constant_mapper = DataLakeKeyDict.CONSTANT_MAPPER
        self.field_name_mapper = DataLakeKeyDict.FIELD_NAME_MAPPER

    @staticmethod
    def get_fold_indices(mode, dataset_info: dict) -> List[int]:
        """
        Get fold indices based on the mode (TRAIN, VALIDATE, TEST).
        """
        if mode == RunMode.TRAIN:
            fold_indices = dataset_info[DatasetInfoKey.TOTAL_FOLD]
            # Remove all validation folds
            for val_fold in dataset_info.get(DatasetInfoKey.VALIDATE_FOLD, []):
                if val_fold in fold_indices:
                    fold_indices.remove(val_fold)
            # Remove all test folds (if any)
            for test_fold in dataset_info.get(DatasetInfoKey.TEST_FOLD, []):
                if test_fold in fold_indices:
                    fold_indices.remove(test_fold)
        elif mode == RunMode.VALIDATE:
            fold_indices = dataset_info.get(DatasetInfoKey.VALIDATE_FOLD, [])
        elif mode == RunMode.TEST:
            fold_indices = dataset_info.get(DatasetInfoKey.TEST_FOLD, [])
        else:
            raise ValueError("Unexpected mode was given.")

        return fold_indices

    @staticmethod
    def fetch_documents(
        collection: str,
        query: Union[dict, omegaconf.DictConfig] = None,
        projection: dict = None,
        field_name: str = None,
        verbose=False,
    ) -> List:
        # Run data lake client.
        with pymongo.MongoClient(DB_ADDRESS) as client:
            # Set query.
            if isinstance(query, dict):
                query = query
            elif isinstance(query, omegaconf.DictConfig):
                query = OmegaConf.to_container(query, resolve=True)
            else:
                query = {}

            # Set projection.
            if projection:
                projection[DataLakeKey.DOC_ID] = 1
            else:
                projection = {}

            # Load documents.
            docs = [
                doc[field_name] if field_name else doc for doc in client[TARGET_DB][collection].find(query, projection)
            ]
            if verbose:
                logger.info(f"Loaded {len(docs)} documents.\nCollection: {collection}\nQuery:\n{query}")

        return docs

    def fetch_multiple_datasets(self, dataset_infos, mode=None) -> pd.DataFrame:
        assert dataset_infos is not None, "dataset_infos is not given."
        dfs = list()
        for dataset, dataset_info in dataset_infos.items():
            # set collection
            assert DatasetInfoKey.COLLECTION_NAME in dataset_info.keys(), "collection name is not given."
            collection_name = dataset_info[DatasetInfoKey.COLLECTION_NAME]

            # set query
            query = (
                OmegaConf.create({})
                if dataset_info[DatasetInfoKey.QUERY] is None
                else dataset_info[DatasetInfoKey.QUERY]
            )
            if mode:
                fold_indices = self.get_fold_indices(mode, dataset_info)
                # the case of luna16 dataset, the fold can be set by subset index instead of fold.
                # the name of key for fold var. is set by "key_fold".
                query[getattr(dataset_info, "key_fold", DatasetInfoKey.FOLD)] = {"$in": fold_indices}

            # get dataframe
            docs = self.fetch_documents(collection=collection_name, query=query)
            df = pd.DataFrame(docs)
            if len(df) != 0:
                # dataset & collection name
                df[DatasetInfoKey.DATASET] = dataset
                if DatasetInfoKey.COLLECTION_NAME in dataset_info.keys():
                    df[DataLakeKey.COLLECTION] = dataset_info[DatasetInfoKey.COLLECTION_NAME]

                # constant_mapper
                if self.constant_mapper in dataset_info.keys():
                    for key, value in dataset_info[self.constant_mapper].items():
                        if key in df:
                            raise ValueError(f"The new feature name '{key}' has already been given.")
                        df[key] = value if value else None

                # field_name_mapper
                # e.g., r_coord_zyx_label: r_coord (dataset A)
                #       r_coord_zyx_label: r_coord_zyx (dataset B)
                if self.field_name_mapper in dataset_info.keys():
                    for key, value in dataset_info[self.field_name_mapper].items():
                        if key in df and key not in _OVERWRITE_ALLOWED:
                            raise ValueError(f"The new feature name '{key}' has already been given.")
                        if isinstance(value, omegaconf.dictconfig.DictConfig):
                            source = value.source
                            field = value.field
                            df[key] = df[source].apply(lambda x: x[field])
                        else:
                            df[key] = df[value] if value else None

                dfs.append(df)

        return pd.concat(dfs, ignore_index=True) if len(dfs) != 0 else pd.DataFrame()

    @staticmethod
    def update_existing_docs(
        df: pd.DataFrame,
        updated_cols: list,
        field_prefix: str,
        doc_id_key: str = DataLakeKey.DOC_ID,
        collection_key: str = DataLakeKey.COLLECTION,
        dbms_uri: str = DB_ADDRESS,
        db: str = TARGET_DB,
    ) -> None:
        """Updates fields of existing documents at DBMS(MongoDB).

        Assumes dataframe has DB fields: (1) document ID and (2) collection name.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to be uploaded as documents.
        updated_cols : list
            Columns selected for upload.
        field_prefix : str
            Prefix added to column name to produce field name.
        doc_id_key : str, optional
            DB key of document ID field, by default DatabaseKey.DOC_ID
        collection_key : str, optional
            DB key of collection field, by default DatabaseKey.COLLECTION
        dbms_uri : str, optional
            URI(address) to database management system, by default c.DB_ADDRESS
        db : str, optional
            Target database name at DBMS, by default c.TARGET_DB
        """

        required_columns = [doc_id_key, collection_key]
        for col in required_columns:
            assert col in df.columns, f"Missing required column '{col}' in input DataFrame."

        # Run database client.
        with pymongo.MongoClient(dbms_uri) as client:
            # Upload each row(i.e., document)
            for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Uploading to DB"):
                # Document field is named as dataframe column name with prefix.
                doc = dict()
                for col in updated_cols:
                    field = f"{field_prefix}_{col}" if field_prefix else col
                    doc[field] = row[col]

                # Prepare arguments for MongoDB update_one()
                doc_id = row[doc_id_key]
                # Below two are used at assert message.
                id_match_filter = {doc_id_key: ObjectId(doc_id)}
                update = {"$set": doc}  # follows MongoDB format

                # Update fields to database.
                collection = row[collection_key]
                update_result = client[db][collection].update_one(id_match_filter, update)
                assert bool(update_result.matched_count), (
                    f"Failed to find matching id.\n"
                    f"Failed case:\n"
                    f" - ID match filter: {id_match_filter}\n"
                    f" - update: {update}\n"
                    f"Raw result from MongoDB:\n"
                    f"{update_result.raw_result}\n"
                )
