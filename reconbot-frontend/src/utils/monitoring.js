// src/utils/monitoring.js - Error Monitoring and Analytics
class SecurityMonitor {
  constructor() {
    this.errorQueue = [];
    this.securityEvents = [];
    this.isProduction = process.env.NODE_ENV === 'production';
  }

  // Log security events
  logSecurityEvent(event, severity = 'info', details = {}) {
    const securityEvent = {
      timestamp: new Date().toISOString(),
      event,
      severity,
      details,
      userAgent: navigator.userAgent,
      url: window.location.href,
      sessionId: this.getSessionId()
    };

    this.securityEvents.push(securityEvent);

    // Send to monitoring service in production
    if (this.isProduction && severity === 'high') {
      this.sendSecurityAlert(securityEvent);
    }

    console.log(`[SECURITY] ${severity.toUpperCase()}: ${event}`, details);
  }

  // Log application errors
  logError(error, errorInfo = {}, context = '') {
    const errorEvent = {
      timestamp: new Date().toISOString(),
      message: error.message,
      stack: error.stack,
      name: error.name,
      context,
      errorInfo,
      url: window.location.href,
      userAgent: navigator.userAgent,
      sessionId: this.getSessionId()
    };

    this.errorQueue.push(errorEvent);

    // Send to error reporting service
    if (this.isProduction) {
      this.sendErrorReport(errorEvent);
    }

    console.error('[ERROR]', errorEvent);
  }

  // Track user interactions for security analysis
  trackInteraction(action, element, data = {}) {
    if (!this.isProduction) return;

    const interaction = {
      timestamp: new Date().toISOString(),
      action,
      element,
      data,
      sessionId: this.getSessionId()
    };

    // Send to analytics service (anonymized)
    this.sendAnalytics(interaction);
  }

  // Send security alerts to monitoring service
  async sendSecurityAlert(event) {
    try {
      await fetch('/api/security/alert', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(event)
      });
    } catch (error) {
      console.error('Failed to send security alert:', error);
    }
  }

  // Send error reports
  async sendErrorReport(errorEvent) {
    try {
      await fetch('/api/errors/report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(errorEvent)
      });
    } catch (error) {
      console.error('Failed to send error report:', error);
    }
  }

  // Send anonymized analytics
  async sendAnalytics(interaction) {
    try {
      // Only send non-sensitive interaction data
      const sanitizedInteraction = {
        timestamp: interaction.timestamp,
        action: interaction.action,
        element: interaction.element,
        sessionId: interaction.sessionId
      };

      await fetch('/api/analytics/track', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(sanitizedInteraction)
      });
    } catch (error) {
      console.error('Failed to send analytics:', error);
    }
  }

  // Get or create session ID
  getSessionId() {
    let sessionId = sessionStorage.getItem('session_id');
    if (!sessionId) {
      sessionId = 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
      sessionStorage.setItem('session_id', sessionId);
    }
    return sessionId;
  }

  // Get current error summary
  getErrorSummary() {
    return {
      totalErrors: this.errorQueue.length,
      recentErrors: this.errorQueue.slice(-10),
      securityEvents: this.securityEvents.slice(-10)
    };
  }
}

// Create global instance
const securityMonitor = new SecurityMonitor();

// Export for use throughout the app
export default securityMonitor;

// Convenience functions
export const logSecurityEvent = (event, severity, details) =>
  securityMonitor.logSecurityEvent(event, severity, details);

export const logError = (error, errorInfo, context) =>
  securityMonitor.logError(error, errorInfo, context);

export const trackInteraction = (action, element, data) =>
  securityMonitor.trackInteraction(action, element, data);