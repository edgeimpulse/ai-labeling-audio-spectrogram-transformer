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


from typing import Dict, List, Optional
from pydantic import BaseModel, Field, StrictStr
from edgeimpulse_api.models.classify_job_response_all_of_accuracy_total_summary import ClassifyJobResponseAllOfAccuracyTotalSummary

class ClassifyJobResponseAllOfAccuracy(BaseModel):
    total_summary: ClassifyJobResponseAllOfAccuracyTotalSummary = Field(..., alias="totalSummary")
    summary_per_class: Dict[str, ClassifyJobResponseAllOfAccuracyTotalSummary] = Field(..., alias="summaryPerClass")
    confusion_matrix_values: Dict[str, Dict[str, float]] = Field(..., alias="confusionMatrixValues")
    all_labels: List[StrictStr] = Field(..., alias="allLabels")
    accuracy_score: Optional[float] = Field(None, alias="accuracyScore")
    mse_score: Optional[float] = Field(None, alias="mseScore")
    __properties = ["totalSummary", "summaryPerClass", "confusionMatrixValues", "allLabels", "accuracyScore", "mseScore"]

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
    def from_json(cls, json_str: str) -> ClassifyJobResponseAllOfAccuracy:
        """Create an instance of ClassifyJobResponseAllOfAccuracy from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of total_summary
        if self.total_summary:
            _dict['totalSummary'] = self.total_summary.to_dict()
        # override the default output from pydantic by calling `to_dict()` of each value in summary_per_class (dict)
        _field_dict = {}
        if self.summary_per_class:
            for _key in self.summary_per_class:
                if self.summary_per_class[_key]:
                    _field_dict[_key] = self.summary_per_class[_key].to_dict()
            _dict['summaryPerClass'] = _field_dict
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> ClassifyJobResponseAllOfAccuracy:
        """Create an instance of ClassifyJobResponseAllOfAccuracy from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return ClassifyJobResponseAllOfAccuracy.construct(**obj)

        _obj = ClassifyJobResponseAllOfAccuracy.construct(**{
            "total_summary": ClassifyJobResponseAllOfAccuracyTotalSummary.from_dict(obj.get("totalSummary")) if obj.get("totalSummary") is not None else None,
            "summary_per_class": dict((_k, Dict[str, ClassifyJobResponseAllOfAccuracyTotalSummary].from_dict(_v)) for _k, _v in obj.get("summaryPerClass").items()),
            "confusion_matrix_values": obj.get("confusionMatrixValues"),
            "all_labels": obj.get("allLabels"),
            "accuracy_score": obj.get("accuracyScore"),
            "mse_score": obj.get("mseScore")
        })
        return _obj
