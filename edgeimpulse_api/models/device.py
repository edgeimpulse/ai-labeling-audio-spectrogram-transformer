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

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, StrictBool, StrictInt, StrictStr, validator
from edgeimpulse_api.models.device_inference_info import DeviceInferenceInfo
from edgeimpulse_api.models.device_sensors_inner import DeviceSensorsInner

class Device(BaseModel):
    id: StrictInt = ...
    device_id: StrictStr = Field(..., alias="deviceId", description="Unique identifier (such as MAC address) for a device")
    created: datetime = ...
    last_seen: datetime = Field(..., alias="lastSeen", description="Last message that was received from the device (ignoring keep-alive)")
    name: StrictStr = ...
    device_type: StrictStr = Field(..., alias="deviceType")
    sensors: List[DeviceSensorsInner] = ...
    remote_mgmt_connected: StrictBool = Field(..., description="Whether the device is connected to the remote management interface. This property is deprecated, use `remoteMgmtMode` instead.")
    remote_mgmt_host: Optional[StrictStr] = Field(None, description="The remote management host that the device is connected to")
    supports_snapshot_streaming: StrictBool = Field(..., alias="supportsSnapshotStreaming")
    remote_mgmt_mode: StrictStr = Field(..., alias="remoteMgmtMode", description="Replaces `remote_mgmt_connected`. Shows whether the device is connected to the remote management interface, and in which mode.")
    inference_info: Optional[DeviceInferenceInfo] = Field(None, alias="inferenceInfo")
    __properties = ["id", "deviceId", "created", "lastSeen", "name", "deviceType", "sensors", "remote_mgmt_connected", "remote_mgmt_host", "supportsSnapshotStreaming", "remoteMgmtMode", "inferenceInfo"]

    @validator('remote_mgmt_mode')
    def remote_mgmt_mode_validate_enum(cls, v):
        if v not in ('disconnected', 'ingestion', 'inference'):
            raise ValueError("must validate the enum values ('disconnected', 'ingestion', 'inference')")
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
    def from_json(cls, json_str: str) -> Device:
        """Create an instance of Device from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of each item in sensors (list)
        _items = []
        if self.sensors:
            for _item in self.sensors:
                if _item:
                    _items.append(_item.to_dict())
            _dict['sensors'] = _items
        # override the default output from pydantic by calling `to_dict()` of inference_info
        if self.inference_info:
            _dict['inferenceInfo'] = self.inference_info.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> Device:
        """Create an instance of Device from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return Device.construct(**obj)

        _obj = Device.construct(**{
            "id": obj.get("id"),
            "device_id": obj.get("deviceId"),
            "created": obj.get("created"),
            "last_seen": obj.get("lastSeen"),
            "name": obj.get("name"),
            "device_type": obj.get("deviceType"),
            "sensors": [DeviceSensorsInner.from_dict(_item) for _item in obj.get("sensors")] if obj.get("sensors") is not None else None,
            "remote_mgmt_connected": obj.get("remote_mgmt_connected"),
            "remote_mgmt_host": obj.get("remote_mgmt_host"),
            "supports_snapshot_streaming": obj.get("supportsSnapshotStreaming"),
            "remote_mgmt_mode": obj.get("remoteMgmtMode"),
            "inference_info": DeviceInferenceInfo.from_dict(obj.get("inferenceInfo")) if obj.get("inferenceInfo") is not None else None
        })
        return _obj
