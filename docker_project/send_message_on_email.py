import win32com.client as win32
outlook = win32.Dispatch('outlook.application')
mail = outlook.CreateItem(0)
mail.To = 'vsevernov@gmail.com'
mail.Subject = '+1 бал за 3 дз'
mail.Body = 'Добрый день! Я смотрю лекции, поэтому за отправку сообщения получаю +1 балл '

mail.Send()