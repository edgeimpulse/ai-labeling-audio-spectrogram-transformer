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


from typing import List, Optional
from pydantic import BaseModel, Field, StrictInt
from edgeimpulse_api.models.cosine_similarity_issue_issues_inner import CosineSimilarityIssueIssuesInner
from edgeimpulse_api.models.sample import Sample

class CosineSimilarityIssue(BaseModel):
    id: StrictInt = Field(..., description="The ID of this sample")
    sample: Optional[Sample] = None
    label: StrictInt = Field(..., description="The label of this sample, in index form")
    issues: List[CosineSimilarityIssueIssuesInner] = Field(..., description="A list of samples that have windows that are symptomatic of this issue.")
    __properties = ["id", "sample", "label", "issues"]

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
    def from_json(cls, json_str: str) -> CosineSimilarityIssue:
        """Create an instance of CosineSimilarityIssue from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of sample
        if self.sample:
            _dict['sample'] = self.sample.to_dict()
        # override the default output from pydantic by calling `to_dict()` of each item in issues (list)
        _items = []
        if self.issues:
            for _item in self.issues:
                if _item:
                    _items.append(_item.to_dict())
            _dict['issues'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> CosineSimilarityIssue:
        """Create an instance of CosineSimilarityIssue from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return CosineSimilarityIssue.construct(**obj)

        _obj = CosineSimilarityIssue.construct(**{
            "id": obj.get("id"),
            "sample": Sample.from_dict(obj.get("sample")) if obj.get("sample") is not None else None,
            "label": obj.get("label"),
            "issues": [CosineSimilarityIssueIssuesInner.from_dict(_item) for _item in obj.get("issues")] if obj.get("issues") is not None else None
        })
        return _obj

