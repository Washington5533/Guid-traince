/**
 * i18n dictionary keys for Training Guardian plugin.
 *
 * The browser half registers this under the 'training-guardian' namespace
 * via ctx.locale.register(NS, { zh, en }).
 */

export type TgKey =
  // Panel
  | 'panel.title'
  | 'panel.noData'
  | 'panel.connecting'
  | 'panel.disconnected'
  | 'panel.close'
  | 'panel.drag'
  // Connection diagnosis
  | 'conn.failed'
  | 'conn.retry'
  | 'conn.retrying'
  | 'conn.advice'
  | 'conn.adviceBusy'
  | 'conn.attempt'
  | 'conn.unauthorized'
  | 'conn.notFound'
  | 'conn.serverError'
  | 'conn.unreachable'
  | 'conn.exhausted'
  | 'conn.unknown'
  | 'conn.hintUnauthorized'
  | 'conn.hintNotFound'
  | 'conn.hintServerError'
  | 'conn.hintUnreachable'
  // Tabs
  | 'tab.overview'
  | 'tab.gpu'
  | 'tab.anomalies'
  | 'tab.decisions'
  | 'tab.arch'
  // Architecture tab
  | 'arch.title'
  | 'arch.noData'
  | 'arch.loading'
  | 'arch.analyzeBtn'
  | 'arch.error'
  | 'arch.viewTreemap'
  | 'arch.viewBackbone'
  | 'arch.colorParams'
  | 'arch.colorFlops'
  | 'arch.bottlenecks'
  | 'arch.bottleneckCount'
  | 'arch.noBottlenecks'
  | 'arch.params'
  | 'arch.flops'
  | 'arch.repeat'
  | 'arch.subModules'
  | 'arch.moduleCount'
  | 'arch.layerCount'
  | 'arch.elapsedMs'
  // Architecture: agent-driven analysis
  | 'arch.agentBtn'
  | 'arch.agentRunning'
  | 'arch.agentDispatched'
  | 'arch.agentNoSession'
  | 'arch.agentUnavailable'
  | 'arch.modelEntryMissing'
  | 'arch.detectedEntry'
  | 'arch.source'
  | 'arch.sourceAgent'
  | 'arch.sourceDirect'
  // Overview tab
  | 'overview.epoch'
  | 'overview.loss'
  | 'overview.accuracy'
  | 'overview.lr'
  | 'overview.step'
  | 'overview.status'
  | 'overview.noMetrics'
  // GPU tab
  | 'gpu.title'
  | 'gpu.utilization'
  | 'gpu.temperature'
  | 'gpu.memory'
  | 'gpu.power'
  | 'gpu.noData'
  // Anomalies tab
  | 'anomalies.title'
  | 'anomalies.none'
  | 'anomalies.type'
  | 'anomalies.description'
  | 'anomalies.time'
  | 'anomalies.severity'
  // Decisions tab
  | 'decisions.title'
  | 'decisions.none'
  | 'decisions.tool'
  | 'decisions.action'
  | 'decisions.source'
  | 'decisions.time'
  | 'decisions.approve'
  | 'decisions.reject'
  // Settings
  | 'settings.title'
  | 'settings.serverUrl'
  | 'settings.serverUrlHint'
  | 'settings.authToken'
  | 'settings.authTokenHint'
  | 'settings.sessionId'
  | 'settings.sessionIdHint'
  | 'settings.autoConnect'
  | 'settings.saved'
  | 'settings.modelEntry'
  | 'settings.modelEntryHint'
  | 'settings.projectDir'
  | 'settings.projectDirHint'
  // Severity
  | 'severity.low'
  | 'severity.medium'
  | 'severity.high'
  | 'severity.critical'
  // Misc
  | 'misc.unknown'
  | 'misc.never'
  | 'misc.error'
  | 'misc.warning'
  | 'misc.info'

const zh: Record<TgKey, string> = {
  'panel.title': 'Training Guardian',
  'panel.noData': '暂无数据，请检查连接',
  'panel.connecting': '连接中...',
  'panel.disconnected': '已断开',
  'panel.close': '关闭',
  'panel.drag': '拖拽移动',
  // Connection diagnosis
  'conn.failed': '连接失败',
  'conn.retry': '重试',
  'conn.retrying': '正在重试...',
  'conn.advice': 'AI 建议',
  'conn.adviceBusy': '获取建议中...',
  'conn.attempt': '尝试',
  'conn.unauthorized': '认证失败',
  'conn.notFound': '服务未找到',
  'conn.serverError': '服务器错误',
  'conn.unreachable': '无法连接',
  'conn.exhausted': '重试用尽',
  'conn.unknown': '未知错误',
  'conn.hintUnauthorized': '请检查鉴权 Token 是否与 --remote-auth 一致',
  'conn.hintNotFound': '请确认服务器地址和端口是否正确',
  'conn.hintServerError': '服务器内部错误，请查看服务器日志',
  'conn.hintUnreachable': '请检查网络连接和防火墙设置',
  // Tabs
  'tab.overview': '概览',
  'tab.gpu': '设备',
  'tab.anomalies': '异常',
  'tab.decisions': '决策',
  'tab.arch': '架构',
  'overview.epoch': 'Epoch',
  'overview.loss': 'Loss',
  'overview.accuracy': 'Accuracy',
  'overview.lr': 'Learning Rate',
  'overview.step': 'Step',
  'overview.status': '状态',
  'overview.noMetrics': '等待指标数据...',
  'gpu.title': 'GPU 设备状态',
  'gpu.utilization': '利用率',
  'gpu.temperature': '温度',
  'gpu.memory': '显存',
  'gpu.power': '功耗',
  'gpu.noData': '暂无 GPU 数据',
  'anomalies.title': '异常事件',
  'anomalies.none': '暂无异常',
  'anomalies.type': '类型',
  'anomalies.description': '描述',
  'anomalies.time': '时间',
  'anomalies.severity': '严重程度',
  'decisions.title': 'Sub-agent 决策',
  'decisions.none': '暂无待审批决策',
  'decisions.tool': '工具',
  'decisions.action': '动作',
  'decisions.source': '来源',
  'decisions.time': '时间',
  'decisions.approve': '批准',
  'decisions.reject': '驳回',
  // Architecture tab
  'arch.title': '模型架构分析',
  'arch.noData': '点击「分析架构」查看模型结构可视化',
  'arch.loading': '分析中...',
  'arch.analyzeBtn': '分析架构',
  'arch.error': '分析失败',
  'arch.viewTreemap': '占比图',
  'arch.viewBackbone': '流水线',
  'arch.colorParams': '按参数',
  'arch.colorFlops': '按 FLOPs',
  'arch.bottlenecks': '瓶颈层',
  'arch.bottleneckCount': '个瓶颈',
  'arch.noBottlenecks': '未检测到瓶颈层',
  'arch.params': '参数量',
  'arch.flops': 'FLOPs',
  'arch.repeat': '重复',
  'arch.subModules': '子模块',
  'arch.moduleCount': '模块数',
  'arch.layerCount': '层数',
  'arch.elapsedMs': '耗时',
  // Architecture: agent-driven analysis
  'arch.agentBtn': 'AI 分析架构',
  'arch.agentRunning': 'AI 正在分析...',
  'arch.agentDispatched': '已发送给 AI',
  'arch.agentNoSession': '无可用会话',
  'arch.agentUnavailable': 'AI 不可用',
  'arch.modelEntryMissing': '请在设置中配置模型入口 (model_entry)',
  'arch.detectedEntry': '已检测到入口',
  'arch.source': '来源',
  'arch.sourceAgent': 'AI 生成',
  'arch.sourceDirect': '直接分析',
  // Settings
  'settings.title': 'Training Guardian',
  'settings.serverUrl': 'Guardian 服务器地址',
  'settings.serverUrlHint': '算力服务器的 RemoteServer 地址，如 http://192.168.1.100:8765',
  'settings.authToken': '鉴权 Token',
  'settings.authTokenHint': '可选，与 --remote-auth 一致',
  'settings.sessionId': '训练会话 ID',
  'settings.sessionIdHint': '留空则自动连接第一个活跃会话',
  'settings.autoConnect': '自动连接',
  'settings.saved': '已保存',
  'settings.modelEntry': '模型入口',
  'settings.modelEntryHint': '格式 module:function，如 scripts.train:build_model',
  'settings.projectDir': '项目目录',
  'settings.projectDirHint': '训练项目根目录的绝对路径',
  // Severity
  'severity.low': '低',
  'severity.medium': '中',
  'severity.high': '高',
  'severity.critical': '紧急',
  'misc.unknown': '未知',
  'misc.never': '从未',
  'misc.error': '错误',
  'misc.warning': '警告',
  'misc.info': '信息',
}

const en: Record<TgKey, string> = {
  'panel.title': 'Training Guardian',
  'panel.noData': 'No data — check connection',
  'panel.connecting': 'Connecting...',
  'panel.disconnected': 'Disconnected',
  'panel.close': 'Close',
  'panel.drag': 'Drag to move',
  // Connection diagnosis
  'conn.failed': 'Connection failed',
  'conn.retry': 'Retry',
  'conn.retrying': 'Retrying...',
  'conn.advice': 'AI Advice',
  'conn.adviceBusy': 'Getting advice...',
  'conn.attempt': 'Attempt',
  'conn.unauthorized': 'Authentication failed',
  'conn.notFound': 'Service not found',
  'conn.serverError': 'Server error',
  'conn.unreachable': 'Cannot reach server',
  'conn.exhausted': 'Retries exhausted',
  'conn.unknown': 'Unknown error',
  'conn.hintUnauthorized': 'Check that the auth token matches --remote-auth',
  'conn.hintNotFound': 'Verify the server address and port',
  'conn.hintServerError': 'Check the server logs for details',
  'conn.hintUnreachable': 'Check your network and firewall settings',
  // Tabs
  'tab.overview': 'Overview',
  'tab.gpu': 'Devices',
  'tab.anomalies': 'Anomalies',
  'tab.decisions': 'Decisions',
  'tab.arch': 'Architecture',
  'overview.epoch': 'Epoch',
  'overview.loss': 'Loss',
  'overview.accuracy': 'Accuracy',
  'overview.lr': 'LR',
  'overview.step': 'Step',
  'overview.status': 'Status',
  'overview.noMetrics': 'Waiting for metrics...',
  'gpu.title': 'GPU Status',
  'gpu.utilization': 'Utilization',
  'gpu.temperature': 'Temperature',
  'gpu.memory': 'VRAM',
  'gpu.power': 'Power',
  'gpu.noData': 'No GPU data',
  'anomalies.title': 'Anomalies',
  'anomalies.none': 'No anomalies',
  'anomalies.type': 'Type',
  'anomalies.description': 'Description',
  'anomalies.time': 'Time',
  'anomalies.severity': 'Severity',
  'decisions.title': 'Sub-agent Decisions',
  'decisions.none': 'No pending decisions',
  'decisions.tool': 'Tool',
  'decisions.action': 'Action',
  'decisions.source': 'Source',
  'decisions.time': 'Time',
  'decisions.approve': 'Approve',
  'decisions.reject': 'Reject',
  // Architecture tab
  'arch.title': 'Architecture Analysis',
  'arch.noData': 'Click "Analyze" to visualize model architecture',
  'arch.loading': 'Analyzing...',
  'arch.analyzeBtn': 'Analyze',
  'arch.error': 'Analysis failed',
  'arch.viewTreemap': 'Treemap',
  'arch.viewBackbone': 'Flow',
  'arch.colorParams': 'By Params',
  'arch.colorFlops': 'By FLOPs',
  'arch.bottlenecks': 'Bottlenecks',
  'arch.bottleneckCount': 'bottlenecks',
  'arch.noBottlenecks': 'No bottlenecks detected',
  'arch.params': 'Parameters',
  'arch.flops': 'FLOPs',
  'arch.repeat': 'Repeat',
  'arch.subModules': 'Sub-modules',
  'arch.moduleCount': 'Modules',
  'arch.layerCount': 'Layers',
  'arch.elapsedMs': 'Elapsed',
  // Architecture: agent-driven analysis
  'arch.agentBtn': 'AI Analyze',
  'arch.agentRunning': 'AI analyzing...',
  'arch.agentDispatched': 'Sent to AI',
  'arch.agentNoSession': 'No session available',
  'arch.agentUnavailable': 'AI unavailable',
  'arch.modelEntryMissing': 'Configure model entry in settings',
  'arch.detectedEntry': 'Entry detected',
  'arch.source': 'Source',
  'arch.sourceAgent': 'AI-generated',
  'arch.sourceDirect': 'Direct analysis',
  // Settings
  'settings.title': 'Training Guardian',
  'settings.serverUrl': 'Guardian Server URL',
  'settings.serverUrlHint': 'e.g. http://192.168.1.100:8765',
  'settings.authToken': 'Auth Token',
  'settings.authTokenHint': 'Optional, must match --remote-auth',
  'settings.sessionId': 'Session ID',
  'settings.sessionIdHint': 'Leave empty to auto-connect to first active session',
  'settings.autoConnect': 'Auto Connect',
  'settings.saved': 'Saved',
  'settings.modelEntry': 'Model Entry',
  'settings.modelEntryHint': 'Format module:function, e.g. scripts.train:build_model',
  'settings.projectDir': 'Project Directory',
  'settings.projectDirHint': 'Absolute path to the training project root',
  // Severity
  'severity.low': 'Low',
  'severity.medium': 'Medium',
  'severity.high': 'High',
  'severity.critical': 'Critical',
  'misc.unknown': 'Unknown',
  'misc.never': 'Never',
  'misc.error': 'Error',
  'misc.warning': 'Warning',
  'misc.info': 'Info',
}

export { zh, en }
