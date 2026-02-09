# app.py
import os
import json
import time
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")
st.title("📊 AI 습관 트래커")

KST = ZoneInfo("Asia/Seoul")

# -----------------------------
# Sidebar: API Keys
# -----------------------------
with st.sidebar:
    st.header("🔑 API 설정")
    openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    weather_api_key = st.text_input("OpenWeatherMap API Key", type="password", placeholder="입력해주세요")
    st.caption("키는 session_state에 저장하지 않으며, 이 브라우저 세션에서만 사용됩니다.")

# -----------------------------
# Helper: API calls
# -----------------------------
def get_weather(city: str, api_key: str):
    """
    OpenWeatherMap current weather
    - 한국어(lang=kr), 섭씨(units=metric)
    - 실패 시 None 반환
    - timeout=10
    """
    if not api_key:
        return None
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "lang": "kr", "units": "metric"}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        return {
            "city": city,
            "desc": (data.get("weather", [{}])[0].get("description") or "").strip(),
            "temp": data.get("main", {}).get("temp"),
            "feels_like": data.get("main", {}).get("feels_like"),
            "humidity": data.get("main", {}).get("humidity"),
            "wind": data.get("wind", {}).get("speed"),
        }
    except Exception:
        return None


def _breed_from_dog_url(image_url: str) -> str:
    """
    Dog CEO image URL 예시:
    https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg
    -> breeds/<breed>/... 에서 breed 추출
    """
    try:
        marker = "/breeds/"
        if marker not in image_url:
            return "Unknown"
        tail = image_url.split(marker, 1)[1]
        breed_segment = tail.split("/", 1)[0]  # e.g. "hound-afghan"
        breed = breed_segment.replace("-", " ").strip()
        # 보기 좋게 Title Case (단, 너무 어색할 수 있어 그대로도 OK)
        return " ".join([w.capitalize() for w in breed.split()])
    except Exception:
        return "Unknown"


def get_dog_image():
    """
    Dog CEO 랜덤 강아지 사진 URL + 품종 반환
    - 실패 시 None 반환
    - timeout=10
    """
    try:
        url = "https://dog.ceo/api/breeds/image/random"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("status") != "success":
            return None
        image_url = data.get("message")
        if not image_url:
            return None
        breed = _breed_from_dog_url(image_url)
        return {"image_url": image_url, "breed": breed}
    except Exception:
        return None


def generate_report(
    habits: dict,
    mood: int,
    achievement_pct: int,
    weather: dict | None,
    dog: dict | None,
    coach_style: str,
    openai_key: str,
):
    """
    습관+기분+날씨+강아지 품종 -> OpenAI로 전달해 리포트 생성
    모델: gpt-5-mini
    - 실패 시 None 반환
    """
    if not openai_key:
        return None

    # 코치 스타일별 시스템 프롬프트
    style_prompts = {
        "스파르타 코치": (
            "너는 엄격하고 직설적인 코치다. 변명은 받아주지 않는다. "
            "팩트 기반으로 칭찬은 짧게, 피드백은 날카롭게. 실행 가능한 지시를 준다."
        ),
        "따뜻한 멘토": (
            "너는 따뜻하고 다정한 멘토다. 공감과 격려를 먼저 주고, "
            "작은 성공을 확대해준다. 부담 없는 다음 행동을 제안한다."
        ),
        "게임 마스터": (
            "너는 RPG 게임 마스터다. 사용자는 모험가/플레이어. "
            "퀘스트, 보상, 레벨업, 아이템 같은 요소로 재미있게 동기부여한다."
        ),
    }

    system_prompt = f"""
{style_prompts.get(coach_style, style_prompts["따뜻한 멘토"])}

반드시 아래 출력 형식을 지켜라(제목/순서/항목명 동일):
1) 컨디션 등급: (S/A/B/C/D 중 하나)
2) 습관 분석: (짧은 문단 + 핵심 포인트 3개 불릿)
3) 날씨 코멘트: (날씨 기반 조언 1~2문장)
4) 내일 미션: (체크박스로 실천 가능한 3개)
5) 오늘의 한마디: (짧고 임팩트 있게 1문장)

한국어로 작성. 과장된 의학 조언 금지. 구체적으로.
""".strip()

    payload = {
        "date": datetime.now(KST).strftime("%Y-%m-%d"),
        "mood": mood,
        "achievement_pct": achievement_pct,
        "habits": habits,
        "weather": weather,
        "dog_breed": (dog or {}).get("breed") if dog else None,
    }

    user_prompt = f"""
다음 사용자의 오늘 데이터로 코칭 리포트를 작성해줘.

[사용자 데이터(JSON)]
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()

    try:
        # OpenAI Python SDK (v1+) 사용
        # pip install openai
        from openai import OpenAI

        client = OpenAI(api_key=openai_key)

        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        return text if text else None
    except Exception:
        return None


# -----------------------------
# Session State: demo data + records
# -----------------------------
def _init_demo_records():
    # 데모용 6일 샘플 데이터 생성 (과하게 랜덤하지 않게 고정 패턴)
    today = datetime.now(KST).date()
    sample = []
    # 6일치: 오늘-6 ~ 오늘-1
    for i in range(6, 0, -1):
        d = today - timedelta(days=i)
        # 간단한 패턴(일별 변동)
        checked = 2 + (i % 4)  # 2~5
        mood = 4 + ((i * 2) % 7)  # 4~10
        sample.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "checked": int(checked),
                "achievement_pct": int(round(checked / 5 * 100)),
                "mood": int(mood),
            }
        )
    return sample


if "records" not in st.session_state:
    st.session_state.records = _init_demo_records()

if "today_saved" not in st.session_state:
    st.session_state.today_saved = False

# -----------------------------
# Check-in UI
# -----------------------------
st.subheader("✅ 오늘의 습관 체크인")

HABITS = [
    ("wake", "🌅", "기상 미션"),
    ("water", "💧", "물 마시기"),
    ("study", "📚", "공부/독서"),
    ("workout", "🏃", "운동하기"),
    ("sleep", "😴", "수면"),
]

cities = [
    "Seoul",
    "Busan",
    "Incheon",
    "Daegu",
    "Daejeon",
    "Gwangju",
    "Ulsan",
    "Suwon",
    "Jeju",
    "Sejong",
]

coach_style = None

# 2열 배치 체크박스 (5개)
colA, colB = st.columns(2)
habit_values = {}

with colA:
    for key, emoji, label in HABITS[:3]:
        habit_values[key] = st.checkbox(f"{emoji} {label}", value=False, key=f"hb_{key}")

with colB:
    for key, emoji, label in HABITS[3:]:
        habit_values[key] = st.checkbox(f"{emoji} {label}", value=False, key=f"hb_{key}")

mood = st.slider("😊 오늘 기분 점수", min_value=1, max_value=10, value=7, step=1)

c1, c2 = st.columns([1, 1])
with c1:
    city = st.selectbox("🌍 도시 선택", cities, index=0)
with c2:
    coach_style = st.radio("🧠 코치 스타일", ["스파르타 코치", "따뜻한 멘토", "게임 마스터"], horizontal=True)

checked_count = sum(1 for v in habit_values.values() if v)
achievement_pct = int(round((checked_count / 5) * 100))

# -----------------------------
# Metrics
# -----------------------------
st.subheader("📈 오늘의 달성률")
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("달성률", f"{achievement_pct}%")
with m2:
    st.metric("달성 습관", f"{checked_count}/5")
with m3:
    st.metric("기분", f"{mood}/10")

# -----------------------------
# Save today's record to session_state
# -----------------------------
today_str = datetime.now(KST).strftime("%Y-%m-%d")

save_col1, save_col2 = st.columns([1, 3])
with save_col1:
    save_today = st.button("📝 오늘 기록 저장", use_container_width=True)
with save_col2:
    if save_today:
        # 기존 오늘 기록이 있으면 업데이트, 없으면 추가
        updated = False
        for rec in st.session_state.records:
            if rec["date"] == today_str:
                rec["checked"] = checked_count
                rec["achievement_pct"] = achievement_pct
                rec["mood"] = mood
                updated = True
                break
        if not updated:
            st.session_state.records.append(
                {
                    "date": today_str,
                    "checked": checked_count,
                    "achievement_pct": achievement_pct,
                    "mood": mood,
                }
            )
        st.session_state.today_saved = True
        st.success("오늘 기록이 저장되었습니다! (세션 기준)")

# -----------------------------
# 7-day bar chart (6 demo + today)
# -----------------------------
st.subheader("🗓️ 최근 7일 추이")

# 차트용 데이터프레임 구성: records에서 최근 7일만
# 오늘 기록이 저장되지 않았더라도, UI 기준 오늘 값을 임시로 포함(요구사항: 6일 샘플 + 오늘 데이터)
records_df = pd.DataFrame(st.session_state.records)

# 오늘 값이 records에 없다면 임시 추가
if not (records_df["date"] == today_str).any():
    temp_today = pd.DataFrame(
        [
            {
                "date": today_str,
                "checked": checked_count,
                "achievement_pct": achievement_pct,
                "mood": mood,
            }
        ]
    )
    records_df = pd.concat([records_df, temp_today], ignore_index=True)

# 최근 7일 정렬
records_df["date"] = pd.to_datetime(records_df["date"])
records_df = records_df.sort_values("date").tail(7)
chart_df = records_df.set_index("date")[["achievement_pct"]]

st.bar_chart(chart_df)

# -----------------------------
# AI Coach Report Section
# -----------------------------
st.subheader("🧾 AI 코치 컨디션 리포트")

btn = st.button("✨ 컨디션 리포트 생성", use_container_width=True)

if btn:
    # Fetch weather + dog
    with st.spinner("날씨/강아지 정보를 가져오고, AI 리포트를 생성 중..."):
        weather = get_weather(city, weather_api_key)
        dog = get_dog_image()

        habits_readable = {
            "기상 미션": bool(habit_values["wake"]),
            "물 마시기": bool(habit_values["water"]),
            "공부/독서": bool(habit_values["study"]),
            "운동하기": bool(habit_values["workout"]),
            "수면": bool(habit_values["sleep"]),
        }

        report = generate_report(
            habits=habits_readable,
            mood=mood,
            achievement_pct=achievement_pct,
            weather=weather,
            dog=dog,
            coach_style=coach_style,
            openai_key=openai_api_key,
        )

    # Layout: weather + dog cards in 2 columns
    card1, card2 = st.columns(2)

    with card1:
        st.markdown("### 🌦️ 오늘의 날씨")
        if weather:
            st.write(f"**도시:** {weather['city']}")
            st.write(f"**상태:** {weather['desc'] or '정보 없음'}")
            st.write(f"**기온:** {weather['temp']}°C (체감 {weather['feels_like']}°C)")
            st.write(f"**습도:** {weather['humidity']}%")
            st.write(f"**바람:** {weather['wind']} m/s")
        else:
            st.warning("날씨 정보를 가져오지 못했어요. (API Key/도시/네트워크 확인)")

    with card2:
        st.markdown("### 🐶 오늘의 강아지")
        if dog:
            st.write(f"**품종:** {dog.get('breed', 'Unknown')}")
            st.image(dog["image_url"], use_container_width=True)
        else:
            st.warning("강아지 이미지를 가져오지 못했어요. (네트워크 확인)")

    st.markdown("---")

    if report:
        st.markdown("### 🧠 AI 코치 리포트")
        st.markdown(report)
    else:
        st.error("AI 리포트를 생성하지 못했어요. (OpenAI API Key/네트워크/할당량 확인)")

    # Share text
    st.markdown("### 📣 공유용 텍스트")
    habit_done = [name for name, done in habits_readable.items() if done]
    habit_not = [name for name, done in habits_readable.items() if not done]
    weather_one = (
        f"{weather['city']} {weather['desc']} / {weather['temp']}°C"
        if weather and weather.get("temp") is not None
        else "날씨 정보 없음"
    )
    dog_one = dog.get("breed") if dog else "강아지 정보 없음"

    share_text = f"""[AI 습관 트래커] {today_str}
- 달성률: {achievement_pct}% ({checked_count}/5)
- 완료: {", ".join(habit_done) if habit_done else "없음"}
- 미완료: {", ".join(habit_not) if habit_not else "없음"}
- 기분: {mood}/10
- 날씨: {weather_one}
- 강아지: {dog_one}

{("—\n" + report) if report else "(리포트 생성 실패)"}
"""
    st.code(share_text)

# -----------------------------
# API 안내 (expander)
# -----------------------------
with st.expander("ℹ️ API 안내 / 준비물"):
    st.markdown(
        """
**필요한 키**
- **OpenAI API Key**: 리포트 생성용 (사이드바에 입력)
- **OpenWeatherMap API Key**: 날씨 표시용 (사이드바에 입력)

**사용한 외부 API**
- OpenWeatherMap (Current Weather): 도시의 현재 날씨(한국어/섭씨)
- Dog CEO API: 랜덤 강아지 이미지

**설치**
```bash
pip install streamlit requests pandas openai

