import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# =====================================================================
# DEMONSTRACJA: CZYM SĄ TOOLS W LANGCHAIN?
# =====================================================================
#
# Jak działają Tools w LangChain? (ekwiwalent Function Calling, ale abstrakcja wyżej)
# 1. Definiujesz narzędzia dekoratorem @tool – LangChain sam generuje ich schemat JSON.
# 2. Przekazujesz je do modelu przez llm.bind_tools(tools).
# 3. Zadajesz pytanie (Prompt).
# 4. Model zwraca AIMessage z listą tool_calls (chce użyć narzędzi).
# 5. Twój kod uruchamia wskazane funkcje i opakowuje wyniki w ToolMessage.
# 6. Wysyłasz całą historię z powrotem – model generuje finalną odpowiedź.
#
# Różnica vs. "surowe" Function Calling z OpenAI SDK:
# - Nie musisz ręcznie pisać schematu JSON { "type": "function", "function": {...} }
# - Dekorator @tool sam wyciąga opis ze docstringa i typy z podpowiedzi typów (type hints)
# - LangChain unifikuje interfejs – ten sam kod zadziała z OpenAI, Anthropic, Gemini itd.
# =====================================================================


# Krok 1: Definiujemy narzędzie dekoratorem @tool
# Docstring staje się opisem dla modelu, type hints → schemat parametrów.
@tool
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Zwraca aktualną pogodę w podanej lokalizacji.

    Args:
        location: Miasto i kraj, np. Warszawa, Polska
        unit: Jednostka temperatury – 'celsius' lub 'fahrenheit'
    """
    # W rzeczywistej aplikacji tutaj znajdowałoby się zapytanie do prawdziwego API pogodowego
    # np. requests.get(f"https://api.weather.com/...&q={location}")
    # Tu posługujemy się atrapą (mock) danych:

    if "Warszawa" in location:
        return f"{{'location': '{location}', 'temperature': 15, 'unit': '{unit}', 'forecast': 'pochmurno'}}"
    elif "Kraków" in location or "Krakow" in location:
        return f"{{'location': '{location}', 'temperature': 18, 'unit': '{unit}', 'forecast': 'słonecznie'}}"
    else:
        return f"{{'location': '{location}', 'temperature': 22, 'unit': '{unit}', 'forecast': 'bezchmurnie'}}"


def main():
    # Krok 2: Inicjalizujemy model z LangChain i "przywiązujemy" do niego narzędzia.
    # bind_tools() automatycznie przekazuje schemat narzędzi przy każdym wywołaniu modelu.
    tools = [get_current_weather]

    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.environ.get("OPENAI_API_KEY", "tutaj_wpisz_swoj_klucz_jesli_nie_masz_w_zm._srodowiskowej"),
    )

    llm_with_tools = llm.bind_tools(tools)

    # Krok 3: Budujemy historię wiadomości i wysyłamy pierwsze zapytanie
    messages = [HumanMessage(content="Cześć! Jaka jest dzisiaj pogoda w Warszawie, a jaka w Krakowie?")]

    print("\n👩 Użytkownik: Cześć! Jaka jest dzisiaj pogoda w Warszawie, a jaka w Krakowie?\n")
    print("🤖 [Etap 1] LLM myśli czy potrzebuje użyć narzędzi...\n")

    # Krok 4: Wywołujemy model – jeśli potrzebuje narzędzi, response.tool_calls będzie niepuste
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # Krok 5: Sprawdzamy, czy model chce użyć narzędzi
    if response.tool_calls:
        print("🛠️  Model LLM zdecydował, że nie ma tych danych w swojej pamięci, więc prosi o wywołanie narzędzia:")

        # Słownik mapujący nazwę narzędzia na jego funkcję Pythona
        available_tools = {t.name: t for t in tools}

        # Iterujemy przez wszystkie żądania wywołań narzędzi (model może żądać kilku naraz!)
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            print(f"   -> Model żąda uruchomienia narzędzia: '{tool_name}' z argumentami: {tool_args}")

            # Uruchamiamy wskazane narzędzie z argumentami od modelu
            tool_fn = available_tools[tool_name]
            tool_result = tool_fn.invoke(tool_args)

            # Wynik opakowujemy w ToolMessage i dodajemy do historii
            messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call_id,
                )
            )

        print("\n🤖 [Etap 2] Zebraliśmy dane. Teraz LLM ponownie generuje ostateczną odpowiedź w języku naturalnym...")

        # Krok 6: Wysyłamy całą historię – model widzi wyniki narzędzi i formułuje odpowiedź
        final_response = llm_with_tools.invoke(messages)

        print("\n✅ Ostateczna odpowiedź modelu do użytkownika:")
        print("--------------------------------------------------")
        print(final_response.content)
        print("--------------------------------------------------")

    else:
        # Model nie potrzebował narzędzi – odpowiedział bezpośrednio
        print("\n✅ Model odpowiedział bez użycia narzędzi:")
        print(response.content)


if __name__ == "__main__":
    main()
