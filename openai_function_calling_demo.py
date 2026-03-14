import json
import os
from openai import OpenAI

# =====================================================================
# DEMONSTRACJA: CZYM JEST FUNCTION CALLING W KONTEKŚCIE LLM?
# =====================================================================
#
# Jak działa Function Calling (wywoływanie funkcji przez LLM)?
# 1. Definiujesz dostępne narzędzia (np. funkcję API pogodowego) i opisujesz algorytmowi
#    jakie przyjmują argumenty, by model LLM wiedział, do czego służą.
# 2. Zadajesz pytanie (Prompt).
# 3. Model zauważa, że odpowiedź wymaga użycia zewnętrznego narzędzia. Zwraca JSON-a 
#    z nazwami funkcji i argumentami do ich wywołania (zamiast gotowego tekstu).
# 4. Twój kod uruchamia te funkcje (np. strzela do prawdziwego API).
# 5. Przekazujesz wynik działania tych funkcji z powrotem do modelu.
# 6. Model analizuje przesłany wynik i na jego podstawie generuje dla Ciebie pełną, 
#    naturalną odpowiedź.
# =====================================================================

# Inicjalizacja klienta OpenAI
# Upewnij się, że masz ustawioną zmienną środowiskową OPENAI_API_KEY
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "tutaj_wpisz_swoj_klucz_jesli_nie_masz_w_zm._srodowiskowej"))

# Krok 1: Definiujemy lokalną funkcję, którą docelowo model "będzie chciał" wywołać
def get_current_weather(location, unit="celsius"):
    """Zwraca aktualną pogodę dla podanej lokalizacji."""
    
    # W rzeczywistej aplikacji tutaj znajdowałoby się zapytanie do prawdziwego API pogodowego
    # np. requests.get(f"https://api.weather.com/...&q={location}")
    # Tu posługujemy się atrapą (mock) danych:
    
    if "Warszawa" in location:
        weather_info = {"location": location, "temperature": 15, "unit": unit, "forecast": "pochmurno"}
    elif "Kraków" in location or "Krakow" in location:
        weather_info = {"location": location, "temperature": 18, "unit": unit, "forecast": "słonecznie"}
    else:
        weather_info = {"location": location, "temperature": 22, "unit": unit, "forecast": "bezchmurnie"}
        
    return json.dumps(weather_info)

def main():
    # Krok 2: Tworzymy schemat (opis) naszej funkcji.
    # Przekazujemy to do modelu, aby wiedział jakie funkcje są dla niego dostępne, 
    # jakie parametry przyjmują i jakiego są one typu.
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Zwraca aktualną pogodę w podanej lokalizacji",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Miasto i stan/kraj, np. Warszawa, Polska",
                        },
                        "unit": {
                            "type": "string", 
                            "enum": ["celsius", "fahrenheit"]
                        },
                    },
                    # Model zawsze będzie wiedział, że musi wyciągnąć "location" z pytania użytkownika
                    "required": ["location"],
                },
            },
        }
    ]

    # Startowa wiadomość od użytkownika
    messages = [
        {"role": "user", "content": "Cześć! Jaka jest dzisiaj pogoda w Warszawie, a jaka w Krakowie?"}
    ]
    
    print("\n👩 Użytkownik: Cześć! Jaka jest dzisiaj pogoda w Warszawie, a jaka w Krakowie?\n")
    print("🤖 [Etap 1] LLM myśli czy potrzebuje użyć narzędzi...\n")

    # Krok 3: Wysyłamy zapytanie do modelu, przekazując jednocześnie listę 'tools'
    response = client.chat.completions.create(
        model="gpt-4o",   # lub inny model np. "gpt-3.5-turbo"
        messages=messages,
        tools=tools,
        tool_choice="auto",  # "auto" oznacza, że to model sam zdecyduje, czy użyć funkcji czy po prostu odpowiedzieć tekstowo
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # Krok 4: Sprawdzamy, czy w odpowiedzi modelu jest w ogóle prośba o wywołanie funkcji
    if tool_calls:
        print("🛠️  Model LLM zdecydował, że nie ma tych danych w swojej pamięci, więc prosi nas o wywołanie zewnętrznej funkcji:")
        
        # Zapisujemy odpowiedź modelu żądającą wywołania funkcji do historii konwersacji (wymóg API)
        messages.append(response_message)
        
        # Słownik, gdzie mapujemy nazwę tekstową z API na prawdziwą funkcję w Pythonie
        available_functions = {
            "get_current_weather": get_current_weather,
        }
        
        # Krok 5: Iterujemy przez żądane przez model wywołania (może ich być kilka na raz, np. dla W-wy i KRK niezależnie!)
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            
            # Argumenty od modelu to ZAWSZE string z JSON-em, należy go sparsować na obiekt Pythona
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"   -> Model żąda uruchomienia funkcji: '{function_name}' z argumentami: {function_args}")
            
            # Uruchamiamy naszą Pythonową funkcję (odpytanie "bazy" czy "zewnętrznego API pogodowego")
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit", "celsius"),
            )
            
            # Po wykonaniu funkcji w Pythonie, formatujemy wynik powrotem jako wiadomość o roli "tool" 
            # i doczepiamy wynik do historii konwersacji.
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )
        
        print("\n🤖 [Etap 2] Zebraliśmy dane. Teraz LLM ponownie generuje ostateczną odpowiedź w języku naturalnym...")
        
        # Krok 6: Wysyłamy do modelu CAŁĄ historię. Użytkownik pyta -> Model prosi o wykonanie funkcji -> My je wykonaliśmy i rzucamy mu wynik
        second_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        
        # Model wykorzystał dane, które mu ściągnęliśmy lokalnie, ubrał je w kulturalne słowa i stworzył ostateczną odpowiedź.
        print("\n✅ Ostateczna odpowiedź modelu do użytkownika:")
        print("--------------------------------------------------")
        print(second_response.choices[0].message.content)
        print("--------------------------------------------------")

if __name__ == "__main__":
    main()
