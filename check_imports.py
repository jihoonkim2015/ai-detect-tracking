"""
DeepSORT import 문제 해결을 위한 헬퍼 스크립트
"""

def test_deepsort_imports():
    """
    다양한 DeepSORT import 방법 테스트
    """
    import_attempts = [
        ("deep_sort_realtime", "DeepSort"),
        ("deep_sort_realtime.deepsort_tracker", "DeepSort"),
        ("deep_sort_realtime.deep_sort", "DeepSort"),
        ("deep_sort_realtime.deep_sort.deep_sort", "DeepSort"),
    ]
    
    successful_import = None
    
    for module_path, class_name in import_attempts:
        try:
            exec(f"from {module_path} import {class_name}")
            print(f"✓ 성공: from {module_path} import {class_name}")
            successful_import = (module_path, class_name)
            break
        except ImportError as e:
            print(f"✗ 실패: from {module_path} import {class_name} - {e}")
    
    if successful_import:
        print(f"\n권장 import: from {successful_import[0]} import {successful_import[1]}")
        return successful_import
    else:
        print("\nDeepSORT를 import할 수 없습니다. 간단한 트래커를 사용합니다.")
        return None

if __name__ == "__main__":
    test_deepsort_imports()
