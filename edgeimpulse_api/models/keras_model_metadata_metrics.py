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


from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, StrictBool, StrictInt, StrictStr, validator
from edgeimpulse_api.models.additional_metric import AdditionalMetric
from edgeimpulse_api.models.keras_model_metadata_metrics_on_device_performance_inner import KerasModelMetadataMetricsOnDevicePerformanceInner
from edgeimpulse_api.models.keras_model_type_enum import KerasModelTypeEnum
from edgeimpulse_api.models.model_prediction import ModelPrediction

class KerasModelMetadataMetrics(BaseModel):
    type: KerasModelTypeEnum = ...
    loss: float = Field(..., description="The model's loss on the validation set after training")
    accuracy: Optional[float] = Field(None, description="The model's accuracy on the validation set after training")
    confusion_matrix: List[List[float]] = Field(..., alias="confusionMatrix")
    report: Dict[str, Any] = Field(..., description="Precision, recall, F1 and support scores")
    on_device_performance: List[KerasModelMetadataMetricsOnDevicePerformanceInner] = Field(..., alias="onDevicePerformance")
    predictions: Optional[List[ModelPrediction]] = None
    visualization: StrictStr = ...
    is_supported_on_mcu: StrictBool = Field(..., alias="isSupportedOnMcu")
    mcu_support_error: Optional[StrictStr] = Field(None, alias="mcuSupportError")
    profiling_job_id: Optional[StrictInt] = Field(None, alias="profilingJobId", description="If this is set, then we're still profiling this model. Subscribe to job updates to see when it's done (afterward the metadata will be updated).")
    profiling_job_failed: Optional[StrictBool] = Field(None, alias="profilingJobFailed", description="If this is set, then the profiling job failed (get the status by getting the job logs for 'profilingJobId').")
    additional_metrics: List[AdditionalMetric] = Field(..., alias="additionalMetrics")
    __properties = ["type", "loss", "accuracy", "confusionMatrix", "report", "onDevicePerformance", "predictions", "visualization", "isSupportedOnMcu", "mcuSupportError", "profilingJobId", "profilingJobFailed", "additionalMetrics"]

    @validator('visualization')
    def visualization_validate_enum(cls, v):
        if v not in ('featureExplorer', 'dataExplorer', 'none'):
            raise ValueError("must validate the enum values ('featureExplorer', 'dataExplorer', 'none')")
        return v

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
    def from_json(cls, json_str: str) -> KerasModelMetadataMetrics:
        """Create an instance of KerasModelMetadataMetrics from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of each item in on_device_performance (list)
        _items = []
        if self.on_device_performance:
            for _item in self.on_device_performance:
                if _item:
                    _items.append(_item.to_dict())
            _dict['onDevicePerformance'] = _items
        # override the default output from pydantic by calling `to_dict()` of each item in predictions (list)
        _items = []
        if self.predictions:
            for _item in self.predictions:
                if _item:
                    _items.append(_item.to_dict())
            _dict['predictions'] = _items
        # override the default output from pydantic by calling `to_dict()` of each item in additional_metrics (list)
        _items = []
        if self.additional_metrics:
            for _item in self.additional_metrics:
                if _item:
                    _items.append(_item.to_dict())
            _dict['additionalMetrics'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> KerasModelMetadataMetrics:
        """Create an instance of KerasModelMetadataMetrics from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return KerasModelMetadataMetrics.construct(**obj)

        _obj = KerasModelMetadataMetrics.construct(**{
            "type": obj.get("type"),
            "loss": obj.get("loss"),
            "accuracy": obj.get("accuracy"),
            "confusion_matrix": obj.get("confusionMatrix"),
            "report": obj.get("report"),
            "on_device_performance": [KerasModelMetadataMetricsOnDevicePerformanceInner.from_dict(_item) for _item in obj.get("onDevicePerformance")] if obj.get("onDevicePerformance") is not None else None,
            "predictions": [ModelPrediction.from_dict(_item) for _item in obj.get("predictions")] if obj.get("predictions") is not None else None,
            "visualization": obj.get("visualization"),
            "is_supported_on_mcu": obj.get("isSupportedOnMcu"),
            "mcu_support_error": obj.get("mcuSupportError"),
            "profiling_job_id": obj.get("profilingJobId"),
            "profiling_job_failed": obj.get("profilingJobFailed"),
            "additional_metrics": [AdditionalMetric.from_dict(_item) for _item in obj.get("additionalMetrics")] if obj.get("additionalMetrics") is not None else None
        })
        return _obj
