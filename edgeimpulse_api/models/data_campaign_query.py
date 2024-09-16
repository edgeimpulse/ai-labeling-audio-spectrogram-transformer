# coding: utf-8

"""
    Edge Impulse API

    No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Generated by: https://openapi-generator.tech
"""


from __future__ import annotations
from inspect import getfullargspec
import pprint
import re  # noqa: F401
import json



from pydantic import BaseModel, StrictStr

class DataCampaignQuery(BaseModel):
    name: StrictStr = ...
    dataset: StrictStr = ...
    query: StrictStr = ...
    __properties = ["name", "dataset", "query"]

    class Config:
        allow_population_by_field_name = True
        validate_assignment = False

    def to_str(self) -> str:
        """Returns the string representation of the model using alias"""
        return pprint.pformat(self.dict(by_alias=True))

    def to_json(self) -> str:
        """Returns the JSON representation of the model using alias"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> DataCampaignQuery:
        """Create an instance of DataCampaignQuery from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DataCampaignQuery:
        """Create an instance of DataCampaignQuery from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return DataCampaignQuery.construct(**obj)

        _obj = DataCampaignQuery.construct(**{
            "name": obj.get("name"),
            "dataset": obj.get("dataset"),
            "query": obj.get("query")
        })
        return _obj
