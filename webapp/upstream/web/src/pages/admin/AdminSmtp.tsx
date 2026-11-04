import { useEffect } from 'react';
import { Button, Form, Input, InputNumber, Message, Switch } from '@arco-design/web-react';
import { api } from '../../api';

export default function AdminSmtp() {
  const [form] = Form.useForm();

  useEffect(() => {
    api.adminSmtp().then((d) => form.setFieldsValue(d)).catch((e) => Message.error(e.message));
  }, []);

  return (
    <div style={{ maxWidth: 520 }}>
      <h2>SMTP 设置</h2>
      <Form
        form={form}
        layout="vertical"
        onSubmit={async (v) => {
          try {
            await api.adminSetSmtp(v);
            Message.success('已保存');
          } catch (e: any) {
            Message.error(e.message);
          }
        }}
      >
        <Form.Item label="启用 SMTP" field="enabled" triggerPropName="checked">
          <Switch />
        </Form.Item>
        <Form.Item label="Host" field="host"><Input /></Form.Item>
        <Form.Item label="Port" field="port"><InputNumber /></Form.Item>
        <Form.Item label="User" field="user"><Input /></Form.Item>
        <Form.Item label="Password" field="password"><Input.Password placeholder="不修改可留 ***" /></Form.Item>
        <Form.Item label="From" field="from_addr"><Input /></Form.Item>
        <Form.Item label="STARTTLS" field="tls" triggerPropName="checked"><Switch /></Form.Item>
        <Button type="primary" htmlType="submit">保存</Button>
      </Form>
    </div>
  );
}
