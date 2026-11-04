import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Button, Form, Input, Message, Space, Typography } from '@arco-design/web-react';
import { api, setToken } from '../api';

export default function AuthPage({ mode }: { mode: 'login' | 'register' }) {
  const nav = useNavigate();
  const [loading, setLoading] = useState(false);
  const [form] = Form.useForm();

  const onSubmit = async () => {
    const v = await form.validate();
    setLoading(true);
    try {
      const res = mode === 'login'
        ? await api.login({ email: v.email, password: v.password })
        : await api.register({ email: v.email, password: v.password, display_name: v.display_name, code: v.code });
      setToken(res.access_token);
      Message.success(mode === 'login' ? '登录成功' : '注册成功');
      nav('/');
    } catch (e: any) {
      Message.error(e.message || '失败');
    } finally {
      setLoading(false);
    }
  };

  const sendCode = async () => {
    const email = form.getFieldValue('email');
    if (!email) return Message.warning('先填写邮箱');
    try {
      await api.sendCode(email);
      Message.success('验证码已发送（未开 SMTP 时见 API 日志）');
    } catch (e: any) {
      Message.error(e.message);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="brand">ATRI LoRA</div>
        <Typography.Paragraph type="secondary" style={{ marginTop: 0 }}>
          {mode === 'login' ? '登录后与亚托莉聊天' : '邮箱注册（支持 SMTP 验证）'}
        </Typography.Paragraph>
        <Form form={form} layout="vertical" onSubmit={onSubmit}>
          <Form.Item label="邮箱" field="email" rules={[{ required: true, type: 'email' }]}>
            <Input placeholder="you@example.com" />
          </Form.Item>
          {mode === 'register' && (
            <>
              <Form.Item label="显示名" field="display_name">
                <Input placeholder="可选" />
              </Form.Item>
              <Form.Item label="验证码" field="code" rules={[{ required: true }]}>
                <Space>
                  <Input placeholder="6 位验证码" style={{ width: 180 }} />
                  <Button onClick={sendCode}>发送验证码</Button>
                </Space>
              </Form.Item>
            </>
          )}
          <Form.Item label="密码" field="password" rules={[{ required: true, minLength: 8 }]}>
            <Input.Password />
          </Form.Item>
          <Button type="primary" htmlType="submit" long loading={loading} style={{ background: 'var(--accent)' }}>
            {mode === 'login' ? '登录' : '注册'}
          </Button>
        </Form>
        <div style={{ marginTop: 16 }}>
          {mode === 'login' ? (
            <Link to="/register">没有账号？去注册</Link>
          ) : (
            <Link to="/login">已有账号？去登录</Link>
          )}
        </div>
      </div>
    </div>
  );
}
