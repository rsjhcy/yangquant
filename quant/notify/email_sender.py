"""
邮件通知模块 — 纯标准库，无需额外安装
支持 QQ邮箱 / 163邮箱 / 其他 SMTP
"""

import smtplib
from datetime import date, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional

from loguru import logger

from quant.config import config as app_config


class EmailSender:
    """邮件发送器

    配置方式（config.yaml）:
        email:
          smtp_server: "smtp.qq.com"
          smtp_port: 465
          sender: "your_email@qq.com"
          password: "your_smtp_auth_code"    # QQ邮箱用授权码，不是登录密码
          receiver: "receiver@qq.com"
    """

    def __init__(self):
        email_cfg = getattr(app_config, "_app_config", None)
        if email_cfg and hasattr(email_cfg, "email"):
            self.cfg = email_cfg.email
        else:
            # 默认值
            from quant.config import EmailConfig
            self.cfg = EmailConfig()

        self.smtp_server = self.cfg.smtp_server
        self.smtp_port = self.cfg.smtp_port
        self.sender = self.cfg.sender
        self.password = self.cfg.password
        self.receiver = self.cfg.receiver
        # Support multiple receivers (comma-separated)
        self.receivers = [r.strip() for r in self.receiver.split(",") if r.strip()] if self.receiver else []

    def send_recommendation(
        self,
        balanced_picks: List[Dict],
        aggressive_picks: List[Dict],
        market_summary: Optional[Dict] = None,
    ) -> bool:
        """发送每日推荐邮件

        Args:
            balanced_picks: 平衡型推荐 [{symbol, name, score, ...}, ...]
            aggressive_picks: 激进型推荐
            market_summary: 市场概况 {index_change, up_count, down_count, ...}

        Returns:
            是否发送成功
        """
        today = date.today()

        subject = f"[羊量推荐] {today} 今日关注"

        html = self._build_html(
            today, balanced_picks, aggressive_picks, market_summary
        )

        return self._send(subject, html)

    def _send(self, subject: str, html: str) -> bool:
        """底层发送（支持多个收件人，逗号分隔）"""
        if not self.sender or not self.password or not self.receivers:
            logger.error(
                "邮件未配置! config.yaml 中设置: sender/password/receiver(多个用逗号分隔)"
            )
            return False

        from email.header import Header
        from email.utils import formataddr

        all_ok = True
        for recipient in self.receivers:
            try:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = Header(subject, "utf-8")
                msg["From"] = formataddr(("羊量量化", self.sender))
                msg["To"] = recipient
                msg.attach(MIMEText(html, "html", "utf-8"))

                if self.smtp_port == 465:
                    with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, timeout=15) as smtp:
                        smtp.login(self.sender, self.password)
                        smtp.sendmail(self.sender, recipient, msg.as_string())
                else:
                    with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=15) as smtp:
                        smtp.starttls()
                        smtp.login(self.sender, self.password)
                        smtp.sendmail(self.sender, recipient, msg.as_string())

                logger.info(f"邮件已发送 → {recipient}")
            except Exception as e:
                logger.error(f"发送到 {recipient} 失败: {e}")
                all_ok = False

        return all_ok

    @staticmethod
    def _sell_plan_html(plan: dict, plan_str: str) -> str:
        """Generate sell plan HTML row"""
        if not plan:
            return ""
        return f"""
            <tr>
                <td style="padding:8px 16px 8px 48px; background:#fef9e7; border-bottom:1px solid #f0f0f0;">
                    <span style="font-size:12px; color:#e67e22;">📉 <b>卖出计划</b></span>
                    <span style="font-size:12px; color:#888; margin-left:8px;">
                        止损: <b style="color:#e74c3c;">{plan.get('stop_loss','-')}</b>
                        &nbsp;|&nbsp; 止盈: <b style="color:#27ae60;">{plan.get('take_profit_1','-')}</b>
                        &nbsp;|&nbsp; 移动止损: <b>{plan.get('trailing_start','-')}</b>
                        &nbsp;|&nbsp; 水下<b>{plan.get('time_stop_days','-')}</b>天平仓
                        &nbsp;|&nbsp; 最长持有: <b>{plan.get('max_hold_days','-')}天</b>
                    </span>
                    <br><span style="font-size:11px; color:#999;">{plan.get('note','')}</span>
                </td>
            </tr>"""

    def _build_html(
        self,
        today: date,
        balanced: List[Dict],
        aggressive: List[Dict],
        market: Optional[Dict] = None,
    ) -> str:
        """生成 HTML 邮件内容"""

        def pick_card(pick: Dict, rank: int) -> str:
            medal = ["🥇", "🥈", "🥉"][rank] if rank < 3 else f"#{rank+1}"
            reason = pick.get("reason", "")
            notes = pick.get("notes", "")

            score = float(pick.get("score", 0))
            mom = float(pick.get("momentum_score", 0))
            trend = float(pick.get("trend_score", 0))
            vol = float(pick.get("volume_score", 0))

            return f"""
            <tr>
                <td style="padding:14px 16px; border-bottom:1px solid #eee;">
                    <span style="font-size:20px;">{medal}</span>
                    <strong style="font-size:16px; margin-left:6px;">{pick.get('symbol','')}</strong>
                    <span style="color:#666; margin-left:8px;">{pick.get('name','')}</span>
                </td>
            </tr>
            <tr>
                <td style="padding:4px 16px 14px 48px; border-bottom:1px solid #f0f0f0;">
                    <table style="font-size:13px; color:#555;">
                        <tr>
                            <td style="padding:2px 8px;">综合得分: <b style="color:#667eea;">{score:.1f}</b></td>
                            <td style="padding:2px 8px;">动量: {mom:.0f}</td>
                            <td style="padding:2px 8px;">趋势: {trend:.0f}</td>
                            <td style="padding:2px 8px;">量价: {vol:.0f}</td>
                        </tr>
                        <tr>
                            <td style="padding:2px 8px;" colspan="2">昨收: <b>{pick.get('close', '-')}</b></td>
                            <td style="padding:2px 8px;" colspan="2">涨幅: <b style="color:{'#e74c3c' if str(pick.get('pct_chg','0')).startswith('+') else '#27ae60' if str(pick.get('pct_chg','0')).startswith('-') else '#555'};">{pick.get('pct_chg','-')}</b></td>
                        </tr>
                        {f'<tr><td style="padding:2px 8px; color:#888;" colspan="4">📌 {reason}</td></tr>' if reason else ''}
                        {f'<tr><td style="padding:2px 8px; color:#e67e22;" colspan="4">⚠ {notes}</td></tr>' if notes else ''}
                    </table>
                    """ + (
                        self._sell_plan_html(pick.get("sell_plan", {}), pick.get("sell_plan_str", ""))
                        if pick.get("sell_plan") else ""
                    ) + f"""
                </td>
            </tr>"""

        picks_html = ""
        if balanced:
            picks_html += """
            <tr><td style="padding:16px 16px 8px; font-size:15px; color:#2c3e50;">
                <strong>【平衡型推荐】</strong><span style="font-size:12px;color:#999;"> — 适合稳健持仓</span>
            </td></tr>"""
            for i, pick in enumerate(balanced):
                picks_html += pick_card(pick, i)

        if aggressive:
            picks_html += """
            <tr><td style="padding:20px 16px 8px; font-size:15px; color:#e74c3c;">
                <strong>【激进型推荐】</strong><span style="font-size:12px;color:#999;"> — 适合追求短期收益</span>
            </td></tr>"""
            for i, pick in enumerate(aggressive):
                picks_html += pick_card(pick, i)

        market_html = ""
        if market:
            market_html = f"""
            <tr><td style="padding:20px 16px 8px; font-size:15px; color:#2c3e50;"><strong>【市场概况】</strong></td></tr>
            <tr><td style="padding:4px 16px 14px 16px;">
                <span style="color:#666;">上证指数: <b>{market.get('sh_index','-')}</b></span> &nbsp;
                <span style="color:#666;">涨跌家数: <b style="color:#e74c3c;">{market.get('up_count','-')}↑</b> / <b style="color:#27ae60;">{market.get('down_count','-')}↓</b></span> &nbsp;
                <span style="color:#666;">涨停: <b>{market.get('limit_up','-')}</b></span>
            </td></tr>"""

        now = datetime.now().strftime("%H:%M")

        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="margin:0; padding:0; font-family:'Microsoft YaHei',Arial,sans-serif; background:#f5f5f5;">
<table width="100%" cellpadding="0" cellspacing="0" style="max-width:600px; margin:20px auto; background:#fff; border-radius:12px; box-shadow:0 2px 12px rgba(0,0,0,0.08);">
    <!-- Header -->
    <tr>
        <td style="padding:28px 24px; background:linear-gradient(135deg,#667eea,#764ba2); border-radius:12px 12px 0 0; text-align:center;">
            <div style="font-size:28px;">🐑</div>
            <div style="font-size:20px; font-weight:bold; color:#fff; margin-top:4px;">羊量每日精选</div>
            <div style="font-size:14px; color:rgba(255,255,255,0.85); margin-top:6px;">
                {today} · 盘前推荐 · 生成于 {now}
            </div>
        </td>
    </tr>
    {market_html}
    {picks_html}
    <!-- Footer -->
    <tr>
        <td style="padding:20px 16px; text-align:center; color:#999; font-size:12px; border-top:1px solid #f0f0f0; margin-top:10px;">
            ⚠️ 以上为AI分析结果，仅供参考，不构成投资建议<br>
            🐑 羊量量化平台 · 自动生成于 {today} {now}
        </td>
    </tr>
</table>
</body></html>"""
