from flask import Flask, render_template, request, session, redirect, url_for
from app.components.retriever import create_qa_chain
from dotenv import load_dotenv
import os
from markupsafe import Markup

# Загружаем переменные окружения из файла .env
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

app = Flask(__name__)
# Создаем устойчивый секретный ключ для шифрования сессий браузера
app.secret_key = os.urandom(24)

# Кастомный фильтр Jinja2 для замены переносов строк \n на HTML-тег <br>
def nl2br(value):
    if not value:
        return ""
    return Markup(str(value).replace("\n", "<br>\n"))

app.jinja_env.filters['nl2br'] = nl2br

# ГЛОБАЛЬНАЯ ИНИЦИАЛИЗАЦИЯ: База данных и модель загружаются ОДИН раз при старте сервера.
# Это исключает повторную загрузку и зависание сайта при каждом клике.
logger = app.logger
logger.info("Initializing QA Chain on startup...")
GLOBAL_QA_CHAIN = create_qa_chain()


@app.route("/", methods=["GET", "POST"])
def index():
    # Если пользователь зашел впервые, создаем пустую историю переписки в сессии
    if "messages" not in session:
        session["messages"] = []

    if request.method == "POST":
        user_input = request.form.get("prompt")

        if user_input:
            # 1. Сохраняем вопрос пользователя в историю сообщений
            messages = session["messages"]
            messages.append({"role": "user", "content": user_input})
            session["messages"] = messages

            try:
                # Проверяем, создалась ли глобальная цепочка на старте приложения
                if GLOBAL_QA_CHAIN is None:
                    raise Exception("QA chain is not initialized. Check LLM or VectorStore configurations.")
                
                # 2. Вызываем цепочку RAG (на выходе получаем чистую СТРОКУ ответа)
                bot_answer = GLOBAL_QA_CHAIN.invoke({"input": user_input})
                
                # Вывод ответа в консоль терминала для контроля со стороны разработчика
                print("\n" + "="*60)
                print(f"🍏 ТЕКСТ ОТВЕТА ОТ МОДЕЛИ:\n{bot_answer}")
                print("="*60 + "\n")

                # Страховочная заглушка на случай пустого или нечитаемого ответа модели
                if not bot_answer or not bot_answer.strip():
                    bot_answer = "The model could not extract an answer from the provided documents."

                # 3. Добавляем текстовый ответ ассистента в историю сообщений
                messages.append({"role": "assistant", "content": bot_answer})
                session["messages"] = messages

            except Exception as e:
                # Если цепочка падает, мы выводим ИМЯ ошибки (например, StopIteration), чтобы локализовать баг
                error_msg = f"Error: {type(e).__name__} - {str(e)}"
                return render_template("index.html", messages=session["messages"], error=error_msg)
            
        # Защита от повторной отправки формы при обычном F5 (Pattern Post/Redirect/Get)
        return redirect(url_for("index"))
        
    return render_template("index.html", messages=session.get("messages", []))


@app.route("/clear")
def clear():
    # Полное удаление куки-сессии переписки для сброса интерфейса до чистого состояния
    session.pop("messages", None)
    return redirect(url_for("index"))


if __name__ == "__main__":
    # Запускаем Flask на альтернативном порту 5050.
    # Включаем debug=True, чтобы детальные ошибки бэкенда сразу отображались в терминале.
    app.run(host="127.0.0.1", port=5050, debug=True, use_reloader=False)
