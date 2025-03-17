from selenium import webdriver
from selenium.webdriver.common.by import By
#制作excel
from openpyxl import Workbook

# 配置 Selenium
chrome_driver_path = r"C:\Users\null\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe"
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # 无界面模式
driver = webdriver.Chrome(options=options)

#等待时间
driver.implicitly_wait(3)

#参数
offset = 0
page = 69 #自己改
num = 0

#创建excel表格
wb = Workbook()
ws = wb.active



    url = f"https://123av.org/cn/search/%E6%A1%A5%E6%9C%AC%E6%9C%89%E8%8F%9C%20%E5%A5%B3%E4%BB%86"
    offset += 30
    # 打开网页
    driver.get(url)

    # 使用 CSS 选择器提取特定单元格的内容
    try:
        for i in range (1, 31):
            # i是第几行
            num+=1
            print(num)
            List = []
            for j in range(2,7):
                target_cell = driver.find_element(By.CSS_SELECTOR, f"https://123av.org/dm18/cn/actresses/%E6%96%B0%E6%9C%89%E8%8F%9C%20%28%E6%A1%A5%E6%9C%AC%E6%9C%89%E8%8F%9C%29?page=1")
                data = target_cell.text.strip()
                print(data)
                List.append(data)
            ws.append(List)

    except:
        print(num)
        print("未找到匹配的元素，请检查 HTML 结构")

# 关闭浏览器
driver.quit()


wb.save("data-finish.xlsx")
print("Excel 文件已生成！")

