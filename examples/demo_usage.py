from pv_audit import AuditAgent, BaseAdapter

# 客户只需要写这个适配器
class MyYOLO(BaseAdapter):
    def predict(self, img):
        # 你的模型逻辑...
        return {"data": [], "conf": 0.85}

agent = AuditAgent(MyYOLO())
# 只要改这行，想测什么维度就测什么维度
report = agent.run(img, dims=["Lux+", "ISO"], lvls=[0, 20, 40])
report.plot_trend("audit.png")