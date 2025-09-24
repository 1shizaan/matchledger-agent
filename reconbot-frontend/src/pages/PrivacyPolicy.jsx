// src/pages/PrivacyPolicy.jsx
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { useState } from 'react';

const PrivacyPolicy = () => {
  const [activeSection, setActiveSection] = useState(null);

  const quickSummary = {
    title: "Your Privacy in 60 Seconds",
    items: [
      { icon: "🔐", text: "We encrypt everything and delete your files within 24 hours" },
      { icon: "🚫", text: "We never sell your data or share it with advertisers" },
      { icon: "✅", text: "You can delete your account and data anytime" },
      { icon: "🎯", text: "We only collect what's needed to make reconciliation work" }
    ]
  };

  const sections = [
    {
      id: 'what-we-collect',
      title: 'What We Collect',
      icon: '📊',
      summary: 'Only the basics needed to make reconciliation work for you.',
      essentials: [
        'Your email address (to sign you in)',
        'Files you upload for reconciliation',
        'Basic usage info (to fix bugs and improve the service)'
      ],
      details: [
        'Account information: Just your email and login credentials',
        'Financial files: Temporarily processed, then automatically deleted',
        'Error reports: Anonymous logs to help us fix issues',
        'Feedback: Beta program comments and support messages',
        'Preferences: Your settings and how you use the app'
      ]
    },
    {
      id: 'how-we-protect',
      title: 'How We Protect Your Data',
      icon: '🛡️',
      summary: 'Bank-level security with automatic file deletion.',
      essentials: [
        'Bank-grade encryption (the same level banks use)',
        'Files automatically deleted after 24 hours',
        'Secure servers with multiple safety layers'
      ],
      details: [
        'Military-grade encryption (AES-256) for all uploads',
        'Secure cloud hosting on Amazon Web Services',
        'No permanent storage of your financial data',
        'Multi-factor authentication protection',
        'Regular security testing by experts',
        'All data transmission is encrypted (HTTPS)',
        'Strict staff access controls'
      ]
    },
    {
      id: 'your-control',
      title: 'Your Control & Rights',
      icon: '��️',
      summary: 'You have complete control over your data.',
      essentials: [
        'Delete your data anytime (we respond in 24 hours)',
        'Download all your reconciliation history',
        'See exactly what data we have about you'
      ],
      details: [
        'Request immediate data deletion',
        'Export your complete reconciliation history',
        'Turn off analytics collection',
        'Get detailed reports of your data',
        'Correct any personal information',
        'Withdraw permission for data processing',
        'Move your data to other services',
        'File complaints with privacy authorities'
      ]
    },
    {
      id: 'data-sharing',
      title: 'Who We Share With',
      icon: '🤝',
      summary: 'Simple answer: We don\'t sell or share your personal data.',
      essentials: [
        'We never sell your data to anyone',
        'No advertising companies get your info',
        'Only shared when legally required or with your permission'
      ],
      details: [
        'With your explicit permission for integrations',
        'With service providers who sign strict privacy agreements',
        'When required by law or court orders',
        'To protect our rights or safety',
        'In case of company merger (with advance notice)',
        'With security researchers (for vulnerability reports only)'
      ]
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      {/* Background Animation */}
      <div className="absolute inset-0 overflow-hidden">
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-purple-400/20 rounded-full"
            animate={{
              x: [Math.random() * window.innerWidth, Math.random() * window.innerWidth],
              y: [Math.random() * window.innerHeight, Math.random() * window.innerHeight],
            }}
            transition={{
              duration: Math.random() * 20 + 10,
              repeat: Infinity,
              repeatType: "reverse",
            }}
            initial={{
              x: Math.random() * window.innerWidth,
              y: Math.random() * window.innerHeight,
            }}
          />
        ))}
      </div>

      <div className="relative z-10 max-w-5xl mx-auto px-4 py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          {/* Header */}
          <div className="mb-8">
            <Link
              to="/"
              className="inline-flex items-center text-purple-300 hover:text-white mb-6 transition-colors group"
            >
              <svg className="w-5 h-5 mr-2 transition-transform group-hover:-translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7"></path>
              </svg>
              Back to Dashboard
            </Link>

            <motion.h1
              className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-white via-purple-200 to-pink-200 bg-clip-text text-transparent"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              Privacy Policy
            </motion.h1>

            <motion.p
              className="text-lg text-white/70 mb-6"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
            >
              We believe privacy should be simple to understand. Here's how we protect your data.
            </motion.p>
          </div>

          {/* Quick Summary */}
          <motion.div
            className="bg-gradient-to-r from-green-500/10 to-blue-500/10 backdrop-blur-xl rounded-2xl border border-green-400/20 p-6 mb-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
          >
            <div className="flex items-center space-x-3 mb-4">
              <span className="text-2xl">⚡</span>
              <h2 className="text-2xl font-semibold text-white">{quickSummary.title}</h2>
              <span className="bg-green-400/20 text-green-300 px-3 py-1 rounded-full text-sm font-medium">
                1 minute read
              </span>
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              {quickSummary.items.map((item, index) => (
                <motion.div
                  key={index}
                  className="flex items-center space-x-3 p-3 bg-white/5 rounded-lg"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.8 + index * 0.1 }}
                >
                  <span className="text-xl">{item.icon}</span>
                  <span className="text-white/90">{item.text}</span>
                </motion.div>
              ))}
            </div>

            <div className="mt-4 text-center">
              <p className="text-green-200 text-sm">
                💡 <strong>Bottom line:</strong> We built MatchLedger to respect your privacy from day one.
              </p>
            </div>
          </motion.div>

          {/* Detailed Sections */}
          <div className="space-y-6">
            {sections.map((section, index) => (
              <motion.div
                key={section.id}
                className="bg-white/5 backdrop-blur-xl rounded-2xl border border-white/10 overflow-hidden"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 + 1 }}
              >
                {/* Section Header - Always Visible */}
                <div className="p-6">
                  <div className="flex items-center space-x-3 mb-4">
                    <span className="text-2xl">{section.icon}</span>
                    <h3 className="text-xl font-semibold text-white">{section.title}</h3>
                  </div>

                  <p className="text-white/80 mb-4">{section.summary}</p>

                  {/* Essential Points - Always Visible */}
                  <div className="space-y-2 mb-4">
                    {section.essentials.map((essential, essentialIndex) => (
                      <div key={essentialIndex} className="flex items-start space-x-3">
                        <span className="text-purple-400 mt-1 text-sm">✓</span>
                        <span className="text-white/90">{essential}</span>
                      </div>
                    ))}
                  </div>

                  {/* Expand Button */}
                  <button
                    onClick={() => setActiveSection(activeSection === section.id ? null : section.id)}
                    className="flex items-center space-x-2 text-purple-300 hover:text-white transition-colors text-sm font-medium"
                  >
                    <span>See all details</span>
                    <motion.svg
                      className="w-4 h-4"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                      animate={{ rotate: activeSection === section.id ? 180 : 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                    </motion.svg>
                  </button>
                </div>

                {/* Expandable Details */}
                <motion.div
                  className="overflow-hidden"
                  initial={false}
                  animate={{
                    height: activeSection === section.id ? 'auto' : 0,
                    opacity: activeSection === section.id ? 1 : 0
                  }}
                  transition={{ duration: 0.3, ease: "easeInOut" }}
                >
                  <div className="px-6 pb-6 border-t border-white/10">
                    <h4 className="text-white font-medium mb-3 mt-4">Complete Details:</h4>
                    <ul className="space-y-2">
                      {section.details.map((detail, detailIndex) => (
                        <motion.li
                          key={detailIndex}
                          className="flex items-start space-x-3 text-white/70"
                          initial={{ opacity: 0, x: -10 }}
                          animate={{
                            opacity: activeSection === section.id ? 1 : 0,
                            x: activeSection === section.id ? 0 : -10
                          }}
                          transition={{ delay: detailIndex * 0.05 }}
                        >
                          <span className="text-purple-400 mt-1 text-xs">•</span>
                          <span>{detail}</span>
                        </motion.li>
                      ))}
                    </ul>
                  </div>
                </motion.div>
              </motion.div>
            ))}
          </div>

          {/* Data Retention & Cookies - Simplified */}
          <motion.div
            className="bg-white/5 backdrop-blur-xl rounded-2xl border border-white/10 p-6 mt-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.5 }}
          >
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <div className="flex items-center space-x-3 mb-3">
                  <span className="text-xl">⏰</span>
                  <h4 className="text-lg font-semibold text-white">How Long We Keep Data</h4>
                </div>
                <ul className="space-y-2 text-white/80 text-sm">
                  <li>• <strong>Your files:</strong> Deleted within 24 hours</li>
                  <li>• <strong>Results:</strong> Available for 30 days</li>
                  <li>• <strong>Account:</strong> Until you delete it</li>
                  <li>• <strong>Usage stats:</strong> Anonymous, 2 years max</li>
                </ul>
              </div>

              <div>
                <div className="flex items-center space-x-3 mb-3">
                  <span className="text-xl">🍪</span>
                  <h4 className="text-lg font-semibold text-white">Cookies We Use</h4>
                </div>
                <ul className="space-y-2 text-white/80 text-sm">
                  <li>• <strong>Login cookies:</strong> To keep you signed in</li>
                  <li>• <strong>Settings:</strong> Remember your preferences</li>
                  <li>• <strong>Analytics:</strong> Anonymous usage data</li>
                  <li>• <strong>No tracking:</strong> We don't follow you around the web</li>
                </ul>
              </div>
            </div>
          </motion.div>

          {/* Contact Section */}
          <motion.div
            className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 backdrop-blur-xl rounded-2xl border border-purple-400/20 p-8 mt-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.7 }}
          >
            <div className="text-center">
              <h2 className="text-2xl font-semibold text-white mb-4">Questions About Your Privacy?</h2>
              <p className="text-white/70 mb-6">
                We're committed to being transparent. If anything is unclear, just ask!
              </p>

              <div className="grid md:grid-cols-2 gap-4 max-w-lg mx-auto">
                <motion.a
                  href="mailto:privacy@matchledger.in"
                  className="bg-purple-500 hover:bg-purple-600 text-white px-6 py-3 rounded-lg font-medium transition-all duration-300 flex items-center justify-center space-x-2"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <span>Email Privacy Team</span>
                </motion.a>

                <Link to="/contact">
                  <motion.button
                    className="w-full bg-white/10 hover:bg-white/20 text-white px-6 py-3 rounded-lg font-medium border border-white/20 transition-all duration-300"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    View FAQ & Support
                  </motion.button>
                </Link>
              </div>
            </div>
          </motion.div>

          {/* Legal Footer */}
          <motion.div
            className="bg-white/5 backdrop-blur-xl rounded-2xl border border-white/10 p-4 mt-6 text-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.9 }}
          >
            <div className="text-sm text-white/60 space-y-1">
              <p><strong>Last Updated:</strong> {new Date().toLocaleDateString()} • <strong>Version:</strong> 2.1</p>
              <p>We'll email you if we make significant changes to this policy.</p>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
};

export default PrivacyPolicy;