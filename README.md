# Kraken AI Trading Bot

Ein lauffähiger **BTC/EUR- oder ETH/EUR-Spot-Bot** für Kraken. Er trainiert eine logistische Regression auf Kurs-, Volumen- und Trendmerkmalen, testet sie auf späteren Kerzen und handelt standardmäßig mit virtuellem Guthaben. Telegram ist die Handy-Fernbedienung. Der Echtgeldmodus ist separat gesperrt.

## Schnellstart am Computer

Python 3.12 installieren. ZIP entpacken und ein Terminal im Ordner `kraken-ai-bot` öffnen:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python bot.py backtest
python bot.py scan
python bot.py run
```

Unter Windows statt `source ...`: `.venv\Scripts\activate`; statt `cp`: `copy .env.example .env`; falls nötig `py -3.12` statt `python3`.

Die `.env` mit einem Texteditor anpassen. Ohne Telegram und ohne API-Schlüssel läuft `python bot.py run` bereits im Paper-Modus; Ausgabe erscheint im Terminal. Der Rechner muss für den Dauerbetrieb eingeschaltet bleiben. `Strg+C` stoppt den Bot.

## Bedienung auf dem Handy per Telegram

1. In Telegram mit **@BotFather** einen Bot anlegen und den Token in der lokalen `.env` bei `TELEGRAM_BOT_TOKEN` eintragen.
2. Dem neuen Bot einmal `/start` schicken.
3. Im Projektordner `python bot.py chat-id` ausführen; die eigene numerische ID bei `TELEGRAM_CHAT_ID` eintragen. Befehle werden aus Sicherheitsgründen **nur im privaten Chat** angenommen, nicht in Gruppen.
4. `python bot.py run` neu starten. Telegram: `/scan` (nur Signal), `/backtest`, `/status`, `/pause`, `/resume`, `/help`.

Nur die eingetragene private Chat-ID darf Befehle auslösen. Pro Telegram-Token nur **einen** laufenden Bot-Prozess verwenden; zwei Prozesse würden sich beim Abrufen der Befehle stören. Den Bot-Token und API-Schlüssel **nie** in GitHub oder in einen Chat posten. Wird ein Token öffentlich, bei BotFather sofort erneuern.

## Dauerbetrieb über Render (kostenpflichtig)

Den Inhalt **dieses Projektordners** in ein privates GitHub-Repository hochladen, so dass `Dockerfile` und `render.yaml` im Repository-Stamm liegen. In Render **New → Blueprint** auswählen und das Repository verbinden. Das Blueprint erstellt einen **Background Worker** mit persistentem Datenträger. Vor dem Anlegen Tarif und Datenträgerkosten prüfen: Render bietet für Background Worker keinen kostenlosen Compute-Tarif. `TELEGRAM_BOT_TOKEN` und `TELEGRAM_CHAT_ID` als geheime Umgebungsvariablen eintragen. `MODE` bleibt `paper`. Die Datenbank liegt auf `/app/data` und übersteht Neustarts. Das Render-Blueprint startet den Bot direkt mit `python bot.py run`; ein Web-Port ist für Background Worker nicht nötig.

Wenn du ausschließlich ein iPhone hast, ist dies der Weg für dauerhaften Telegram-Betrieb; lokale Ausführung auf einem eigenen Computer ist die Alternative ohne Hostingtarif. Das ZIP ist der fertige Projektcode; ohne laufenden Prozess kann ein Telegram-Bot keine Signale oder Trades ausführen.

## Wie Signale und Paper-Trades funktionieren

- Stündliche oder vierstündliche Kraken-Kerzen; die unfertige aktuelle Kerze wird ausgeschlossen. Pro Anfrage gibt es maximal 720 Kerzen.
- Merkmale: Kursrenditen, gleitende Durchschnitte, Schwankung, Kerzenkörper, Spanne, Volumen und RSI. Die Zielvariable wird nur aus späteren Preisen der historischen Trainingskerzen gebildet.
- Training bei jedem Scan auf den verfügbaren abgeschlossenen Kerzen. Kauf ab 58 % Modellwahrscheinlichkeit; Verkauf bei höchstens 45 %, bei 4 % Rückgang oder 8 % Anstieg vom Einstieg. Der Bot hält maximal eine eigene Spot-Position, ohne Hebel.
- Standard: 25 EUR pro Einstieg, maximal 50 EUR Positionslimit, 1.000 EUR virtuelles Startguthaben. Gebührenannahme 0,4 % und Schlupfannahme 0,1 % pro Order. Einstellungen in `.env` ändern.
- Der historische Test beginnt mit den ersten 70 % als Training und bewertet die späteren 30 % Kerze für Kerze. Vor jeder Testkerze wird das Modell ausschließlich mit den bis dahin bekannten älteren Daten neu trainiert. Er nutzt den eingestellten Einsatz sowie dieselben Kauf-, Verkauf-, Stop- und Gewinnschwellen und rechnet Ausführungen zum nächsten Kerzenbeginn. Er zeigt Rendite, Vergleich mit Halten, maximalen Rückgang und Anzahl der Ausführungen. Börsen-Mindestorders, verfügbare Liquidität und die tatsächliche Ausführung werden vereinfacht; der Modellscore ist **keine** verlässliche Gewinnchance.
- Der laufende Bot verarbeitet jede abgeschlossene Kerze nur einmal. SQLite speichert Paper-Guthaben, Bot-Position und Trades. `/pause` lässt eine offene Position liegen; der Stop-Loss ist eine Software-Regel, kein an der Börse liegender Schutzauftrag. Bei Ausfall oder schnellem Kurssturz kann die tatsächliche Ausführung deutlich schlechter sein.

## Echtgeldmodus: erst nach eigenem Paper-Test

Der Code enthält eine Kraken-Spot-Anbindung, aber es wurde **kein** Echtgeldauftrag mit einem Nutzerkonto getestet. Für eine bewusst gewählte Aktivierung lokal in `.env` setzen:

```dotenv
MODE=live
ENABLE_LIVE_TRADING=NO
```

1. In Kraken Guthaben, offene und geschlossene Orders sowie bestehende Positionen prüfen. Falls dieser Bot vorher schon live gehandelt hat, **keine neue Zustandsdatei anlegen**: zuerst den alten persistenten Datenträger wiederherstellen oder Konto und Zustand manuell abgleichen.
2. Einmalig im Projektordner `python bot.py init-live` ausführen. Das erstellt nur eine neue Zustandsdatei und sendet **keine** Order. Bei bereits vorhandener Datei bricht der Befehl ab. `python bot.py status` zeigt den Zustand auch ohne API-Schlüssel.
3. Danach einen **eigenen API-Schlüssel nur für diesen Bot** erstellen: Rechte für Guthabenabfrage, Orders anlegen und offene/geschlossene Orders abfragen; **keine Auszahlungsrechte**. Schlüssel lokal in `.env` eintragen und erst dann `ENABLE_LIVE_TRADING=YES` setzen. Vor `python bot.py run` Einsatz, Gebühren, verfügbares EUR-Guthaben und Mindestorder prüfen.

Live-Käufe geben ein Euro-Budget vor; Kauf und Verkauf fordern Gebühren in EUR an. Live-Modus nutzt Marktorders, die vom angezeigten Kurs abweichen können. Bestehende manuelle Bestände werden nicht absichtlich verkauft; die Bot-Position wird getrennt gespeichert. Die Schlüssel bleiben nur in `.env` beziehungsweise in geheimen Host-Variablen.

Auf Render kann `init-live` über **Shell** im laufenden Background Worker ausgeführt werden: `MODE=live DATA_DIR=/app/data python bot.py init-live`. Erst danach `MODE`, `ENABLE_LIVE_TRADING` und die geheimen Schlüssel für den Worker setzen und neu starten. Der persistente Datenträger muss derselbe bleiben. Das mitgelieferte Blueprint startet absichtlich mit `MODE=paper`; für einen späteren Live-Betrieb muss diese Einstellung auch im Blueprint angepasst werden.

Falls Kraken eine Order annimmt, ihre Rückmeldung aber unklar ist, sperrt der Bot weitere Trades (`Ungeklärte Order`). **Nicht einfach die Datenbank löschen oder denselben Auftrag erneut absenden**: zuerst in Kraken offene/geschlossene Orders und Guthaben mit der gespeicherten Position abgleichen. Für eine unsichere Order gibt es absichtlich keinen Telegram-Befehl zum Entsperren. Bei Neustarts den dauerhaften Datenträger verwenden; ohne ihn kann eine Live-Position nicht zuverlässig zugeordnet werden.

## Einstellungen und Grenzen

`.env.example` enthält alle Optionen. Nur `BTC/EUR` und `ETH/EUR`, Zeitrahmen `1h` oder `4h`. Der Bot fragt jede Minute nach einer neu abgeschlossenen Kerze. Die Strategie kann Verlust machen und häufig gar kein Kaufsignal liefern. Ein 720-Kerzen-Test ist kurz; erst über längere Zeit Paper-Trades beobachten. Das Bot-Konto und die Bot-Position gelten pro Handelspaar und Modus. **Nur einen Prozess je Paar und Modus** starten.

## Tests

```bash
python -m unittest discover -s tests -v
```

Die Tests benötigen kein Konto. Ein echter API-Abruf und Echtgeldorders wurden in der Erstellungsumgebung nicht durchgeführt.
