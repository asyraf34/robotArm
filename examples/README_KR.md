# 예제 빠른 가이드

아래는 사용하게 될 핵심 함수들, 입력값, 그리고 반환값입니다. 코드 작성 시 참고하세요.

## 시뮬레이터 연결

- `connect_robot(host="localhost", port=8008", scene_path="scenes/demo.json", verbose=True)` → `RobotController`

  - 컨트롤러를 생성하고 서버에서 해당 scene을 불러옵니다.
  - 보통 scene만 바꾸고 나머지는 기본값을 사용하시면 됩니다.

- 컨텍스트 매니저:
  `with RobotController() as robot:`
  자동으로 연결 및 연결 해제를 수행합니다.

## 이동 (Movement)

- `robot.move_to([x, y], elbow="up")` → `bool`
  - 엔드이펙터(end-effector)를 `[x, y]` 위치로 이동합니다. 성공 시 `True` 반환.
  - elbow 파라미터로 로봇 팔꿈치 방향을 지정할 수 있습니다 ("up" 또는 "down"). 기본 "up"사용을 권장합니다.

- `robot.execute_waypoints([[x1, y1], ...])` → `int`

  - 웨이포인트 리스트를 순차 실행합니다. 도달한 웨이포인트 개수를 반환합니다.

- `robot.line_motion(start_xy, end_xy, n_points=8)` → `int`

  - `n_points`로 보간된 직선 경로를 따라 이동합니다.

- `robot.reset()` → `bool`

  - 로봇을 홈 포지션 `[0, 0, 0, 0]`으로 되돌립니다.
  - 모든 미션을 끝낸 후 초기 위치로 돌아갈 때 유용합니다.

## 미션 (Missions)

- `robot.start_mission("Mission N")` → `bool`
- `robot.complete_mission("Mission N")` → `bool`

  - 매 미션의 시작과 완료 시 이 함수를 사용하세요. 미션 이름은 scene JSON에 정의된 대로 사용해야 합니다.

## 씬 및 상태 (Scene and State)

- `robot.state()` → `dict`

  - 현재 조인트 각도와 엔드이펙터 위치를 반환합니다.
- `robot.position()` → `[x, y]`

  - 엔드이펙터의 현재 위치만 반환합니다.

## 일반적인 패턴 (Typical Pattern)

```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from client_tools import RobotController

with RobotController() as robot:
    robot.load_scene("scenes/easy_scene.json")
    robot.start_mission("Mission 1")
    robot.move_to([0.46, 0.12])
    robot.move_to([0.22, -0.30])
    robot.complete_mission("Mission 1")
```

모든 타이밍/애니메이션은 서버에서 처리하므로 `sleep`이나 지연 코드를 넣을 필요가 없습니다.
