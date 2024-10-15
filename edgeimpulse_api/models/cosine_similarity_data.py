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


from typing import List
from pydantic import BaseModel, Field
from edgeimpulse_api.models.cosine_similarity_issue import CosineSimilarityIssue

class CosineSimilarityData(BaseModel):
    similar_but_different_label: List[CosineSimilarityIssue] = Field(..., alias="similarButDifferentLabel", description="A list of samples that have windows that are similar to windows of other samples that have a different label.")
    different_but_same_label: List[CosineSimilarityIssue] = Field(..., alias="differentButSameLabel", description="A list of samples that have windows that are dissimilar to windows of other samples that have the same label.")
    __properties = ["similarButDifferentLabel", "differentButSameLabel"]

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
    def from_json(cls, json_str: str) -> CosineSimilarityData:
        """Create an instance of CosineSimilarityData from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of each item in similar_but_different_label (list)
        _items = []
        if self.similar_but_different_label:
            for _item in self.similar_but_different_label:
                if _item:
                    _items.append(_item.to_dict())
            _dict['similarButDifferentLabel'] = _items
        # override the default output from pydantic by calling `to_dict()` of each item in different_but_same_label (list)
        _items = []
        if self.different_but_same_label:
            for _item in self.different_but_same_label:
                if _item:
                    _items.append(_item.to_dict())
            _dict['differentButSameLabel'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> CosineSimilarityData:
        """Create an instance of CosineSimilarityData from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return CosineSimilarityData.construct(**obj)

        _obj = CosineSimilarityData.construct(**{
            "similar_but_different_label": [CosineSimilarityIssue.from_dict(_item) for _item in obj.get("similarButDifferentLabel")] if obj.get("similarButDifferentLabel") is not None else None,
            "different_but_same_label": [CosineSimilarityIssue.from_dict(_item) for _item in obj.get("differentButSameLabel")] if obj.get("differentButSameLabel") is not None else None
        })
        return _obj
