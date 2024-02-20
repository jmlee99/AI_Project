# 🔬 Wafer FDC(Fault Detection & Classification) AI_Project
***Create wafer quality inspection classifier AI model with defect data of semiconductor wafers***

성균관대학교 반도체 공정실습을 바탕으로 미래의 공정 데이터는 더욱 고도화 된다고 생각했습니다.
<br />
이에 다채널을 위한 드라이버와 FDC의 고도화의 필요성을 느끼고, 프로젝트를 진행하게 되었습니다.
<br />
데이터는 캐글의 WM-811K데이터를 사용했으며, **캐글의 대부분의 사람들이 특정 Shape만 가지고 모델을 구현**했지만
<br />
**저는 모든 Shape을 다 사용하여 모델을 구현해보았습니다.**

---
## 2. 🔬 Wafer FDC(Fault Detection & Classification)
> 반도체 공정 실습에서 Fault Detection & Classification의 고도화의 필요성을 느끼고 진행한 개인 프로젝트
> - 개발기간 : 2023.06.16 ~ 2023.07.14
> - 핵심역할 : 개인, CNN을 사용한 반도체 고성능 FDC 개발
>> - Language : Python3
>> - Skill : numpy, pandas, seaborn, tensorflow, keras,scikit-learn
>> - Tool : Jupyter Notebook
---

<br />

<div align="center">

**1. 데이터 이상치 확인 및 제거**

 데이터의 대략적인 분포를 확인하고, 이상치 제거를 위한 시각적 확인

![스크린샷 2024-01-28 03-30-43](https://github.com/jmlee99/AI_Project/assets/98507134/8b09c64a-aacc-442d-b7f8-ad874edf9248)

**1-1. CNN 모델을 위한 데이터 정제**

 가장 큰 Wafer의 크기에 맞춰 수동 패딩 알고리즘을 만들었습니다.

 방법1, 방법2 는 모두 (노트북 메모리 문제) 커널이 터졌지만,
 
 많은 방법을 고민하다가 방법3으로 모든 Wafer의 크기를 맞추었습니다.

![스크린샷 2024-01-28 03-43-04](https://github.com/jmlee99/AI_Project/assets/98507134/773a789d-e8b7-4768-a437-5b7f3ed89d1f)

 이후 바뀐 Shape의 Data를 pickle파일에 저장 후 EDA를 진행했습니다.

**2. EDA**

 9개의 패턴 시각적 확인

![스크린샷 2024-01-28 04-19-26](https://github.com/jmlee99/AI_Project/assets/98507134/58f45460-c52c-4047-9951-87211f6b8471)


**2-1. Wafer shape별 failure_type 확인**

 특정 Wafer Shape에서 특정 failure_type가 나온다면 공정의 문제가 아니라 Wafer의 문제를 의심해보아야한다.

![스크린샷 2024-01-28 04-21-11](https://github.com/jmlee99/AI_Project/assets/98507134/81e1c4b5-f7ca-44ae-b5f3-1c87ad6ac6fe)

![스크린샷 2024-01-28 04-23-20](https://github.com/jmlee99/AI_Project/assets/98507134/56485818-e9ad-418c-bbc3-0f78498addaf)

***(25, 27), (39, 37)의 Wafer들은 공정의 문제보다 Wafer의 문제를 의심해 보아야한다는 결론***

**3. 모델 구현**

 다양한 모델 비교

![스크린샷 2023-07-21 14-38-47](https://github.com/jmlee99/AI_Project/assets/98507134/6a4643c8-f2c0-43e7-a891-cbee530292a5)

**Test Data의 정확도 대략 97% 정도의 수치를 보인다.**
<br/>
<br/>
<br/>
<br/>
**4. 데이터 분류**

 **실제 태깅 되지 않은(결측 패턴 표시가 없는)데이터를 가지고 FDC를 사용하였을 때 얼마나 잘 분류해 내는지 확인해보겠습니다.**

**4-1. "Center"**

![스크린샷 2024-01-28 04-34-40](https://github.com/jmlee99/AI_Project/assets/98507134/bbf9c5c1-7691-4965-ab23-c37a47bcbc65)

**4-2. "Donut"**

![스크린샷 2024-01-28 04-35-31](https://github.com/jmlee99/AI_Project/assets/98507134/c67ac0e3-3749-4271-a5d0-e324721a9044)

**4-3. Edge-Loc**

![스크린샷 2024-01-28 04-36-28](https://github.com/jmlee99/AI_Project/assets/98507134/883ba673-2e54-4e26-a40f-998468b1771a)


**4-4. Edge-Ring**

![스크린샷 2024-01-28 04-37-33](https://github.com/jmlee99/AI_Project/assets/98507134/90c60f82-2353-4db8-b3a7-57ac821d2f71)

**4-5. Loc**

![스크린샷 2024-01-28 04-37-59](https://github.com/jmlee99/AI_Project/assets/98507134/8cfde7eb-335a-47e8-a189-9a26fe941d08)

**4-6. Near-Full**

![스크린샷 2024-01-28 04-38-26](https://github.com/jmlee99/AI_Project/assets/98507134/22af0ef2-437b-4cba-aa92-abac180a0721)

**4-7. Random**

![스크린샷 2024-01-28 04-39-13](https://github.com/jmlee99/AI_Project/assets/98507134/936b6f72-f8ee-4629-aa5a-27c9468b3202)

**4-8. Scratch**

![스크린샷 2024-01-28 04-39-44](https://github.com/jmlee99/AI_Project/assets/98507134/d7a03f54-a4c2-4db4-a30b-561ce31f1fea)

**4-9. None**

![스크린샷 2024-01-28 04-40-45](https://github.com/jmlee99/AI_Project/assets/98507134/fbaef96e-167b-4848-8593-8af26c3ea2a0)

***다음과 같이 대부분 모든 패턴을 잘 인식하고 분류해 내는 것을 알 수 있습니다.***


**5. 고찰**

좋은 성능의 FDC를 만들었다고 생각하지만 효율이 좋다고 생각해보면 아닌 것 같다는 생각이 든다.

파라미터의 크기가 너무 커서 공정에서 실시간 검출에 사용되기에는 속도가 많이 느리고 제약이 많을 것으로 생각된다.

이러한 부분은 고쳐 나가야 하는 부분인 것 같다.

또한 수동패딩을 진행함에 있어 2darray를 Flatten 시켜 1차원을 패딩을 주고 Conv1D 모델을 생성하는 것도 흥미로울것 같다.

</div>
