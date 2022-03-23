---
layout: post
title:  "DEEPAS_v1.1 Release Note"
date:   2022-03-28 09:00:00 +0900
categories: deepas
---

<meta charset="utf-8">

- metric이 정확히 나오지 않는 문제가 해결되었습니다. 
- `check_na`와 `fill_na`가 `read_data`에 통합되었습니다. `read_data`의 파라미터로 nan관련 옵션을 입력하는 방식으로 변경되었습니다.
- `tune_model`, `create_model`, `FeatureImportance`에서 `device`, `gpu_id` 파라미터를 더이상 요구하지 않도록 변경되었습니다.
- XAI 기법 사용 시, 어떤 기법을 사용했는지 파일명에 명시하도록 수정되었습니다.
- 하이퍼파라미터 튜닝 결과로 생성된 파일의 명이 기존의 `tuning_result.csv` 에서 `params_info.csv`로 수정되었습니다.
- 하이퍼파라미터 튜닝을 하지 않을 경우, 사용자가 직접 입력한 하이퍼파라미터에 대한 정보를 저장할 수 있도록 수정되었습니다.
- 클래스별 metric 파일 생성 시, 클래스의 이름을 파일명에 명시하도록 수정되었습니다.
- 리눅스 명령창에서 옵션을 주는 방식으로 실행 가능한 방식이 추가되었습니다. 기존의 방식과 명령창 방식 모두 가능합니다. 자세한 실행 방법에 대하여 [DEEPAS-v1.1 Documentation]을 참고해주시기 바랍니다.

[DEEPAS-v1.1 Documentation]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html 