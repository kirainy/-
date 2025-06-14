# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 10:41:39 2024

@author: 86187
"""

tools_all = [
    {
        "type": "function",
        "function": {
            "name": "get_table_data",
            "description": "根据数据表名、开始时间、结束时间、列名和状态获取指定时间范围内的相关数据。返回值为包含指定列名和对应值的字典。",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "数据表名，例如 'device_logs'。",
                    },
                    "start_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    "end_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的结束时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 12:00:00'。",
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "需要查询的列名列表。如果未提供，则返回所有列。",
                        "default": [],
                    },
                    "status": {
                        "type": "string",
                        "description": "需要筛选的状态（例如 'A架开机'、'A架关机'、'小艇入水'、'征服者起吊'、‘征服者入水’等动作）。如果未提供，则不筛选状态。",
                        "default": "",
                    },
                },
                "required": ["table_name", "start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_total_energy",
            "description": "计算指定时间段内指定设备的总能耗。返回值为总能耗（kWh，float 类型）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    "end_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的结束时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 12:00:00'。",
                    },
                    "device_name": {
                        "type": "string",
                        "description": "设备名称，支持以下值：'折臂吊车'、'一号门架'、'二号门架'、'绞车'、'一号发电机'、'二号发电机'、'三号发电机'、'四号发电机'、'一号推进变频器'、'二号推进变频器'、'侧推'、'可伸缩推'、'一号舵桨转舵A'、'一号舵桨转舵B'、'二号舵桨转舵A'、'二号舵桨转舵B'。",
                        "enum": ["折臂吊车", "一号门架", "二号门架", "绞车", "一号发电机", "二号发电机", "三号发电机", "四号发电机", "一号推进变频器", "二号推进变频器", "侧推", "可伸缩推","一号舵桨转舵A","一号舵桨转舵B","二号舵桨转舵A","二号舵桨转舵B"],
                    },
                },
                "required": ["start_time", "end_time", "device_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_total_fuel_volume",
            "description": "计算指定时间段内一号或二号或三号或四号的燃油消耗量。返回值为总能耗（L，float 类型）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    "end_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的结束时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 12:00:00'。",
                    },
                    "device_name": {
                        "type": "string",
                        "description": "设备名称，支持以下值：'一号发电机'、'二号发电机'、'三号发电机'、'四号发电机'。",
                        "enum": ["一号发电机", "二号发电机", "三号发电机", "四号发电机"],
                    },
                },
                "required": ["start_time", "end_time", "device_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_total_deck_machinery_energy",
            "description": "计算甲板机械在指定时间范围内的总能耗。返回值为总能耗（kWh，float 类型）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    "end_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的结束时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 12:00:00'。",
                    },
                },
                "required": ["start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_tuojiang_energy",
            "description": "计算舵桨在指定时间范围内的总能耗。返回值为总能耗（kWh，float 类型）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    "end_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的结束时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 12:00:00'。",
                    },
                },
                "required": ["start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_total_propulsion_system_energy",
            "description": "计算推进系统（所有推进器，包括一号主推、二号主推、侧推和可伸缩推）在指定时间范围内的总能耗。返回值为总能耗（kWh，float 类型）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    "end_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的结束时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 12:00:00'。",
                    },
                },
                "required": ["start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_ajia_energy",
            "description": "计算A架在指定时间范围内的总能耗。返回值为总能耗（kWh，float 类型）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    "end_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的结束时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 12:00:00'。",
                    },
                },
                "required": ["start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_electricity_generation",
            "description": "计算指定时间段内指定某个发电机的总能耗。返回值为总能耗（kWh，float 类型）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    "end_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的结束时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 12:00:00'。",
                    },
                    "device_name": {
                        "type": "string",
                        "description": "设备名称，支持以下值：'一号发电机'、'二号发电机'、'三号发电机'、'四号发电机'。",
                        "enum": ["一号发电机", "二号发电机", "三号发电机", "四号发电机"],
                    },
                },
                "required": ["start_time", "end_time", "device_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_total_electricity_generation",
            "description": "计算总发电量，即一号、二号、三号和四号柴油发电机组的总发电量。返回值为发电量（kWh，float 类型）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    "end_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的结束时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 12:00:00'。",
                    },
                },
                "required": ["start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_total_4_fuel_volume",
            "description": "计算总燃油消耗量，即一至四号柴油发电机组总燃油消耗量。返回值为燃油消耗量（L，float 类型）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    "end_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的结束时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 12:00:00'。",
                    },
                },
                "required": ["start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_device_parameter",
            "description": "通过参数中文名查询该设备参数信息，返回包含该设备参数信息的字典,如报警值等,以便于对比问题中设备的字段与参数信息做对比",
       
            "parameters": {
                "type": "object",
                "properties": {
                    "parameter_name_cn": {
                        "type": "string",
                        "description": "参数中文名，用于查询设备参数信息。",
                    }
                },
                "required": ["parameter_name_cn"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_device_status_by_time_range",
            "description": "根据开始时间和结束时间，以及需要查询的动作名称，查询这段时间之内动作发生的时间，返回进行的动作以及时间点",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    "end_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的结束时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 12:00:00'。",
                    },
                    "shebeiname": {
                        "type": "string",
                        "description": "需要查询的动作名称,如‘A架开机’",
                    }
                },
                "required": ["start_time", "end_time","shebeiname"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_device_status_by_time_range",
            "description": "根据开始时间和结束时间，查询这段时间内A架、折臂吊车和定位系统的所有设备状态，返回设备进行的动作以及时间点",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    "end_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的结束时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 12:00:00'。",
                    }
                },
                "required": ["start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_zuoye_stage_by_time_range",
            "description": "根据开始时间和结束时间，查询这段时间内作业的所有布放阶段（过程）、回收阶段（过程）的变化时间点，包括‘布放开始’、‘布放结束’、‘回收开始’、‘回收结束’，返回作业的阶段（过程）变化以及时间点",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    "end_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的结束时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 12:00:00'。",
                    }
                },
                "required": ["start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_new4_stage_by_time_range",
            "description": "根据开始时间和结束时间，查询这段时间内的状态变化，包括‘动力定位’、‘航渡’、‘伴航’、‘停泊’，返回状态变化以及时间点",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    "end_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的结束时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 12:00:00'。",
                    },
                    "stage_name": {
                        "type": "string",
                        "enum": ["动力定位","航渡","伴航","停泊"],
                        "description": "查询的状态的名称，包括‘动力定位’、‘航渡’、‘伴航’、‘停泊’",
                        "default": "动力定位",
                    },
                },
                "required": ["start_time", "end_time","stage_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_uptime",
            "description": "计算指定时间段内指定设备的总的开机时长或运行时长(注意,这里不是实际开机时长和实际运行时长)，或者计算某个设备由开到关的过程，以及计算这个时间段内的开关次数，并返回开机时长或运行时长（分钟，int 类型）。设备名称支持 '折臂吊车'、'A架' 、'DP' 、'一号柴油发电机' 、'二号柴油发电机' 、'三号柴油发电机' 、'四号柴油发电机'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    "end_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的结束时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 12:00:00'。",
                    },
                    "shebeiname": {
                        "type": "string",
                        "enum": ["折臂吊车", "A架", "DP","一号柴油发电机","二号柴油发电机","三号柴油发电机","四号柴油发电机"],
                        "description": "设备名称，仅支持 '折臂吊车'、'A架'、'DP'、'一号柴油发电机' 、'二号柴油发电机' 、'三号柴油发电机' 、'四号柴油发电机'，默认为 '折臂吊车'。",
                        "default": "折臂吊车",
                    },
                },
                "required": ["start_time", "end_time","shebeiname"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_start_end",
            "description": "计算指定时间段内设备(仅支持A架开关机、折臂吊车开关机、DP)的开始时间和结束时间的时间对，无法只查询开始或只查询结束，支持：A架开关机、折臂吊车开关机以及ON_DP和OFF_DP,若有多个时间段则返回多个时间段元组，格式为[[start1,end1],[start2,end2]...]，设备名称支持：'折臂吊车'、'A架' 和 'DP'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    "end_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的结束时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 12:00:00'。",
                    },
                    "shebeiname": {
                        "type": "string",
                        "enum": ["折臂吊车", "A架", "DP"],
                        "description": "设备名称，支持 '折臂吊车'、'A架' 和 'DP'，默认为 '折臂吊车'。",
                        "default": "折臂吊车",
                    },
                },
                "required": ["start_time", "end_time","shebeiname"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_actual_operational_duration",
            "description": "计算指定时间段内设备的实际运行时长或实际开机时长，并返回实际运行时长（分钟， int 类型）。设备名称支持 'A架'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    "end_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的结束时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 12:00:00'。",
                    },
                    "device_name": {
                        "type": "string",
                        "enum": ["A架"],
                        "description": "设备名称，支持 'A架'，默认为 'A架'。",
                        "default": "A架",
                    },
                },
                "required": ["start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_max_or_average",
            "description": "计算指定时间段内表的某字段的平均值或最大值或最小值。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    "end_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的结束时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 12:00:00'。",
                    },
                    "table_name": {
                        "type": "string",
                        "description": "需要求数据所在的表名，例如 'device_1_2_meter_102'。",
                        "default": "",
                    },
                    "column_name": {
                        "type": "string",
                        "description": "需要求的字段名，如'1-2-0_v'等。",
                        "default": "",
                    },
                    "max_or_average_or_min": {
                        "type": "string",
                        "description": "需要求的是什么，包括：'max'、'average'、'min'",
                        "default": "max",
                    },
                },
                "required": ["start_time", "end_time", "table_name", "column_name", "max_or_average_or_min"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_sum_data",
            "description": "计算指定表格的一定时间内的数据有多少条,返回条数",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    "end_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的结束时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 12:00:00'。",
                    },
                    "table_name": {
                        "type": "string",
                        "description": "需要求数据所在的表名，例如 'device_1_2_meter_102'。",
                        "default": "",
                    },
                },
                "required": ["start_time", "end_time", "table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_swings_10",
            "description": "计算一段时间内A架摆动次数",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    "end_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的结束时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 12:00:00'。",
                    },
                },
                "required": ["start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_swings",
            "description": "计算一段时间内A架的完整摆动次数",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    "end_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的结束时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 12:00:00'。",
                    },
                },
                "required": ["start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rt_temp_or_press_parameter",
            "description": "找到一至四号柴油发电机所有与温度或压力相关的数值类型参数的值的设备参数列表，不需要进一步筛选，支持设备：'一号柴油发电机','二号柴油发电机','三号柴油发电机','四号柴油发电机'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "temp_or_press": {
                        "type": "string",
                        "description": "需要求与温度相关的参数还是与压力相关的参数，包括中有press和temp两种类型。",
                        "default": "press",
                    },
                    "device_name": {
                        "type": "string",
                        "description": "设备名称，包括'一号柴油发电机','二号柴油发电机','三号柴油发电机','四号柴油发电机'",
                        "default": "二号柴油发电机",
                    },
                    
                },
                "required": ["temp_or_press","device_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "action_judgment",
            "description": "综合分析设备在指定时段的动作序列。根据布放/回收阶段特征进行智能判定，同时整合小艇必现动作数据。返回包含动作时序列表、设备状态变更及操作建议的复合型分析结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        # "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    "end_time": {
                        "type": "string",
                        # "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    "stage": {
                        "type": "string",
                        "description": "深海作业A的阶段名称。上午为下放阶段，下午为回收阶段",
                        "enum": ["布放", "回收"],
                    },
                    "time_of_action": {
                        "type": "string",
                        # "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                },
                "required": ["start_time", "end_time", "stage"],
            },
        },
    },
    {
            "type": "function",
            "function": {
                "name": "natural_processing",
                "description": "纯自然语言处理步骤，无需调用工具，若需要调用其他工具则禁止使用",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instruction": {"type": "string"}
                    }
                }
            }
    },
    {
        "type": "function",
        "function": {
            "name": "get_value_of_time",
            "description": "计算在指定时间点的指定表的指定字段的值,一次只能指定一个时间点(time_point)、一个表名(table_name)和一个列名(column_name)",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_point": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的时间，注意参数名为time_point,格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    },
                    "table_name": {
                        "type": "string",
                        "description": "需要求数据所在的表名，例如 'device_1_2_meter_102'，注意参数名为table_name。",
                        "default": "",
                    },
                    "column_name": {
                        "type": "string",
                        "description": "需要求数据值所在的字段名，如'1-2-0_v'等，注意参数名为column_name。",
                        "default": "",
                    },
                },
                "required": ["time_point", "table_name", "column_name"],
            },
        },
        {
        "type": "function",
        "function": {
            "name": "review_calculate",
            "description": "执行四则运算或计算两个时间点差值。数学模式需提供运算符和两个数字；时间模式需提供两个时间字符串及其格式",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation_type": {
                        "type": "string",
                        "enum": ["math", "time"],
                        "description": "计算模式选择：math-四则运算，time-时间差计算"
                    },
                    "operator": {
                        "type": "string",
                        "enum": ["+", "-", "*", "/", "%"],
                        "description": "运算符（仅math模式需要）"
                    },
                    "num1": {
                        "type": "number",
                        "description": "第一个操作数（仅math模式需要）"
                    },
                    "num2": {
                        "type": "number",
                        "description": "第二个操作数（仅math模式需要）"
                    },
                    "time_str1": {
                        "type": "string",
                        "description": "起始时间字符串（如'2023-10-01 08:30'，仅time模式需要）"
                    },
                    "time_str2": {
                        "type": "string",
                        "description": "结束时间字符串（如'2023-10-02 14:45'，仅time模式需要）"
                    },
                    "time_format": {
                        "type": "string",
                        "description": "时间格式说明符（如'%Y-%m-%d %H:%M'，仅time模式需要）"
                    }
                },
                "required": ["operation_type"],
                "oneOf": [
                    {
                        "required": ["operator", "num1", "num2"],
                        "description": "数学运算必填参数集"
                    },
                    {
                        "required": ["time_str1", "time_str2", "time_format"],
                        "description": "时间计算必填参数集"
                    }
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "is_open",
            "description": "判断设备是否在运行，支持设备：‘一号柴油发电机’,‘二号柴油发电机’,‘三号柴油发电机’,‘四号柴油发电机’,‘侧推’。一次只能指定一个时间点(time_point)、一个设备名(shebeiname)，返回(bool类型)",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_point": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的时间，注意参数名为time_point,格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    },
                    "shebeiname": {
                        "type": "string",
                        "description": "需要查的设备名，支持设备：‘一号柴油发电机’,‘二号柴油发电机’,‘三号柴油发电机’,‘四号柴油发电机’,‘侧推’。",
                        "default": "",
                    },
                },
                "required": ["time_point", "shebeiname"],
            },
        },
        {
        "type": "function",
        "function": {
            "name": "count_shoufanglan",
            "description": "计算一段时间内设备的收揽和放缆的次数，支持‘绞车A’，‘绞车B’，‘绞车C’",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 00:00:00'。",
                    },
                    "end_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "查询的结束时间，格式为 'YYYY-MM-DD HH:MM:SS'，例如 '2024-08-23 12:00:00'。",
                    },
                    "shebeiname": {
                        "type": "string",
                        "description": "需要查的设备名，支持设备：‘绞车A’，‘绞车B’，‘绞车C。",
                        "default": "",
                    },
                },
                "required": ["start_time", "end_time","shebeiname"],
            },
        },
    },
]
