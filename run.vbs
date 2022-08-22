Set oShell = WScript.CreateObject ("WScript.Shell")
oShell.run "cmd.exe /C streamlit run app.py"
Set oShell = Nothing'