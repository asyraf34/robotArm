# SCARA 로봇 시뮬레이터

4-DOF SCARA 로봇을 위한 경량 GUI + 클라이언트/서버 기반 시뮬레이터.

- 정/역기구학, 충돌 검사, 직선 경로 및 RRT 기반 플래너
- 미션 리스트 및 애니메이션을 포함한 내장 GUI
- 실행 가능한 예제와 함께 제공되는 간단한 Python 클라이언트 API

> Windows에서 테스트됨. 다른 플랫폼은 보장되지 않음.

## 빠른 설치 (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## GUI 실행

```bash
python run_gui.py
```

`scenes/` 폴더(easy/medium/hard/demo)에서 원하는 씬을 선택하고
미션 탭을 이용해 진행 상황을 확인할 수 있습니다.

## 예제 클라이언트 실행

```bash
python examples/01_basic_movement.py
```

추가 예제는 `examples/` 폴더에 있습니다 (미션, 경로, 충돌 데모 등).

## 씬 구성

`scenes/` 내 JSON 파일을 수정해
장애물, 타깃, 미션을 정의할 수 있습니다.
좌표를 조정하여 새로운 레이아웃을 만들 수 있습니다.

## 도움이 필요하신가요?

샘플에서 사용된 클라이언트 함수들은
`examples/README.md`에서 간단한 가이드를 확인할 수 있습니다.
