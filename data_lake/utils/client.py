import pymongo

from data_lake.constants import DB_ADDRESS


def get_client():
    return pymongo.MongoClient(DB_ADDRESS)
