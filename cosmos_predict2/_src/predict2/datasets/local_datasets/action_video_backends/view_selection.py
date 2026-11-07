"""Pure view selection extracted from OminiData/multi_view_utils.py."""
from typing import List, Optional

PRIMARY_VIEW_PRIORITY = ['camera_top', 'camera_front', 'camera_front_external', 'camera_center', 'camera_head', 'top', 'front', 'center', 'head']

LEFT_VIEW_KEYWORDS = ['left', 'camera_left', 'camera_left_external']

RIGHT_VIEW_KEYWORDS = ['right', 'camera_right', 'camera_right_external']

def select_primary_view(camera_poses: List[str], priority_list: Optional[List[str]]=None) -> List[str]:
    """
    自动选择主视角
    
    选择策略：
    1. 按优先级列表查找单个主视角
    2. 如果没有找到，检查是否有左右视角可以拼接
    3. 最后使用第一个可用视角
    
    Args:
        camera_poses: 可用的相机视角列表
        priority_list: 优先级列表（默认使用 PRIMARY_VIEW_PRIORITY）
        
    Returns:
        主视角名称列表（单个或多个用于拼接）
    """
    if not camera_poses:
        return []
    if priority_list is None:
        priority_list = PRIMARY_VIEW_PRIORITY
    for priority_key in priority_list:
        for pose in camera_poses:
            if priority_key in pose.lower():
                return [pose]
    left_view = None
    right_view = None
    for pose in camera_poses:
        pose_lower = pose.lower()
        if any((kw in pose_lower for kw in LEFT_VIEW_KEYWORDS)) and left_view is None:
            left_view = pose
        if any((kw in pose_lower for kw in RIGHT_VIEW_KEYWORDS)) and right_view is None:
            right_view = pose
    if left_view and right_view:
        return [left_view, right_view]
    if left_view:
        return [left_view]
    if right_view:
        return [right_view]
    return [camera_poses[0]]
