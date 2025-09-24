// src/pages/TermsOfService.jsx
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { useState } from 'react';

const TermsOfService = () => {
  const [expandedSection, setExpandedSection] = useState(null);

  const quickSummary = {
    title: "Terms Summary (1 minute read)",
    items: [
      { icon: "🆓", text: "This is free during beta - we may charge later with notice" },
      { icon: "📁", text: "Upload only legitimate financial data you have rights to" },
      { icon: "🧪", text: "Beta means the service might change or have occasional issues" },
      { icon: "⚖️", text: "We're not responsible for your business decisions based on our results" },
      { icon: "❌", text: "You can cancel anytime, we can too with 30 days notice" },
      { icon: "💬", text: "Questions? Email legal@matchledger.in" }
    ]
  };

  const sections = [
    {
      id: 'beta-program',
      title: 'Beta Program Rules',
      icon: '🧪',
      summary: 'You\'re testing our service for free while we improve it.',
      keyPoints: [
        'Free access during beta testing period',
        'Service may change, improve, or have temporary issues',
        'We might ask for your feedback to help us improve'
      ],
      details: [
        'Beta access is completely free for testing and evaluation',
        'Features, availability, and performance may change without notice',
        'We may temporarily pause service for updates or maintenance',
        'Your feedback helps us build a better product',
        'No service level guarantees during beta (we\'re still perfecting things)',
        'Usage limits may apply to ensure fair access for all beta users',
        'Beta period ends with 30 days notice to all users'
      ]
    },
    {
      id: 'what-you-can-do',
      title: 'What You Can Do',
      icon: '✅',
      summary: 'Use MatchLedger responsibly for legitimate financial reconciliation.',
      keyPoints: [
        'Upload your legitimate financial data for reconciliation',
        'Download and use your reconciliation results',
        'Provide feedback to help us improve the service'
      ],
      details: [
        'Upload financial files you have permission to process',
        'Use our reconciliation tools for legitimate business purposes',
        'Download your results and reconciliation reports',
        'Share feedback about features, bugs, or improvements',
        'Invite colleagues to try the beta (if they qualify)',
        'Contact support when you need help'
      ]
    },
    {
      id: 'what-you-cant-do',
      title: 'What You Can\'t Do',
      icon: '❌',
      summary: 'Don\'t misuse the service or try to break it.',
      keyPoints: [
        'No illegal, fraudulent, or suspicious financial data',
        'Don\'t try to hack, break, or reverse engineer our service',
        'Don\'t share your login credentials with others'
      ],
      details: [
        'Upload files containing illegal or fraudulent transactions',
        'Attempt to hack, reverse engineer, or compromise our service',
        'Share your account credentials or bypass security measures',
        'Upload files that exceed our size, format, or content limits',
        'Use our service to analyze competitors or build competing products',
        'Send viruses, malware, or harmful code through our platform',
        'Provide false information during registration',
        'Try to overwhelm our servers or disrupt service for others'
      ]
    },
    {
      id: 'your-responsibilities',
      title: 'Your Responsibilities',
      icon: '📋',
      summary: 'You\'re responsible for your data and how you use our results.',
      keyPoints: [
        'Make sure you have permission to upload all financial data',
        'Keep backups of important files (we delete them after 24 hours)',
        'Double-check all reconciliation results before making business decisions'
      ],
      details: [
        'Ensure you have the right to upload and process all financial data',
        'Verify uploaded files comply with your organization\'s policies',
        'Maintain backup copies of all important financial records',
        'Review and validate all reconciliation results independently',
        'Avoid uploading unnecessary personally identifiable information',
        'Ensure compliance with your industry\'s data protection rules',
        'Report any security concerns or data breaches immediately',
        'Use results as assistance, not as final business decisions'
      ]
    },
    {
      id: 'our-limitations',
      title: 'What We\'re Not Responsible For',
      icon: '⚖️',
      summary: 'We provide tools to help, but final decisions and accuracy are your responsibility.',
      keyPoints: [
        'Business or financial decisions you make based on our results',
        'Temporary service interruptions (this is beta software)',
        'Complete accuracy of reconciliation results (always double-check)'
      ],
      details: [
        'Our service is provided "as is" during the beta period without warranties',
        'All reconciliation results should be verified by qualified professionals',
        'We\'re not liable for financial decisions made using our system output',
        'Users must ensure compliance with accounting standards and regulations',
        'We don\'t guarantee 100% accuracy in reconciliation results',
        'Service interruptions may occur during development and testing',
        'We\'re not responsible for data loss (though we have strong safeguards)',
        'Maximum liability is limited to the amount paid for services (currently $0)'
      ]
    },
    {
      id: 'account-management',
      title: 'Accounts & Cancellation',
      icon: '🔐',
      summary: 'You can cancel anytime. We can too, but we\'ll give you notice.',
      keyPoints: [
        'Cancel your account anytime through your settings',
        'We\'ll delete all your data within 30 days of cancellation',
        'We may suspend accounts that break these rules'
      ],
      details: [
        'You may delete your account anytime through account settings',
        'We may suspend accounts that violate terms or pose security risks',
        'Account cancellation results in deletion of all data within 30 days',
        'We provide advance notice before account termination when possible',
        'Download any important data before closing your account',
        'Deleted accounts and data cannot be recovered after 30 days',
        'We reserve the right to refuse service for valid reasons'
      ]
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white">
      {/* Background Animation */}
      <div className="absolute inset-0 overflow-hidden">
        {[...Array(25)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-blue-400/20 rounded-full"
            animate={{
              x: [Math.random() * window.innerWidth, Math.random() * window.innerWidth],
              y: [Math.random() * window.innerHeight, Math.random() * window.innerHeight],
            }}
            transition={{
              duration: Math.random() * 25 + 15,
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
              className="inline-flex items-center text-blue-300 hover:text-white mb-6 transition-colors group"
            >
              <svg className="w-5 h-5 mr-2 transition-transform group-hover:-translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7"></path>
              </svg>
              Back to Dashboard
            </Link>

            <motion.h1
              className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-white via-blue-200 to-purple-200 bg-clip-text text-transparent"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              Terms of Service
            </motion.h1>

            <motion.p
              className="text-lg text-white/70 mb-6"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
            >
              Simple, fair terms for using MatchLedger AI. No hidden surprises or confusing legal language.
            </motion.p>
          </div>

          {/* Quick Summary */}
          <motion.div
            className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 backdrop-blur-xl rounded-2xl border border-blue-400/20 p-6 mb-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
          >
            <div className="flex items-center space-x-3 mb-4">
              <span className="text-2xl">⚡</span>
              <h2 className="text-2xl font-semibold text-white">{quickSummary.title}</h2>
              <span className="bg-blue-400/20 text-blue-300 px-3 py-1 rounded-full text-sm font-medium">
                Essential points
              </span>
            </div>

            <div className="grid md:grid-cols-2 gap-3">
              {quickSummary.items.map((item, index) => (
                <motion.div
                  key={index}
                  className="flex items-start space-x-3 p-3 bg-white/5 rounded-lg"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.8 + index * 0.1 }}
                >
                  <span className="text-lg mt-0.5">{item.icon}</span>
                  <span className="text-white/90 text-sm">{item.text}</span>
                </motion.div>
              ))}
            </div>

            <div className="mt-4 text-center">
              <p className="text-blue-200 text-sm">
                💡 <strong>TL;DR:</strong> Use it responsibly, we'll provide a good service, and we can both walk away anytime.
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

                  {/* Key Points - Always Visible */}
                  <div className="space-y-2 mb-4">
                    {section.keyPoints.map((point, pointIndex) => (
                      <div key={pointIndex} className="flex items-start space-x-3">
                        <span className="text-blue-400 mt-1 text-sm">✓</span>
                        <span className="text-white/90">{point}</span>
                      </div>
                    ))}
                  </div>

                  {/* Expand Button */}
                  <button
                    onClick={() => setExpandedSection(expandedSection === section.id ? null : section.id)}
                    className="flex items-center space-x-2 text-blue-300 hover:text-white transition-colors text-sm font-medium"
                  >
                    <span>See complete details</span>
                    <motion.svg
                      className="w-4 h-4"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                      animate={{ rotate: expandedSection === section.id ? 180 : 0 }}
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
                    height: expandedSection === section.id ? 'auto' : 0,
                    opacity: expandedSection === section.id ? 1 : 0
                  }}
                  transition={{ duration: 0.3, ease: "easeInOut" }}
                >
                  <div className="px-6 pb-6 border-t border-white/10">
                    <h4 className="text-white font-medium mb-3 mt-4">Full Details:</h4>
                    <ul className="space-y-2">
                      {section.details.map((detail, detailIndex) => (
                        <motion.li
                          key={detailIndex}
                          className="flex items-start space-x-3 text-white/70"
                          initial={{ opacity: 0, x: -10 }}
                          animate={{
                            opacity: expandedSection === section.id ? 1 : 0,
                            x: expandedSection === section.id ? 0 : -10
                          }}
                          transition={{ delay: detailIndex * 0.05 }}
                        >
                          <span className="text-blue-400 mt-1 text-xs">•</span>
                          <span>{detail}</span>
                        </motion.li>
                      ))}
                    </ul>
                  </div>
                </motion.div>
              </motion.div>
            ))}
          </div>

          {/* Important Legal Notes */}
          <motion.div
            className="bg-white/5 backdrop-blur-xl rounded-2xl border border-white/10 p-6 mt-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.5 }}
          >
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <div className="flex items-center space-x-3 mb-3">
                  <span className="text-xl">📜</span>
                  <h4 className="text-lg font-semibold text-white">Legal Basics</h4>
                </div>
                <ul className="space-y-2 text-white/80 text-sm">
                  <li>• <strong>Jurisdiction:</strong> These terms follow Indian law</li>
                  <li>• <strong>Disputes:</strong> We'll try to resolve issues directly first</li>
                  <li>• <strong>Updates:</strong> We'll email you about major changes</li>
                  <li>• <strong>Validity:</strong> If one part is invalid, the rest still applies</li>
                </ul>
              </div>

              <div>
                <div className="flex items-center space-x-3 mb-3">
                  <span className="text-xl">🔄</span>
                  <h4 className="text-lg font-semibold text-white">Changes & Updates</h4>
                </div>
                <ul className="space-y-2 text-white/80 text-sm">
                  <li>• <strong>Service updates:</strong> New features added regularly</li>
                  <li>• <strong>Terms changes:</strong> 30 days notice for major changes</li>
                  <li>• <strong>Emergency fixes:</strong> Security updates happen immediately</li>
                  <li>• <strong>Your choice:</strong> Don't like changes? You can leave anytime</li>
                </ul>
              </div>
            </div>
          </motion.div>

          {/* Contact Section */}
          <motion.div
            className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 backdrop-blur-xl rounded-2xl border border-blue-400/20 p-8 mt-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.7 }}
          >
            <div className="text-center">
              <h2 className="text-2xl font-semibold text-white mb-4">Questions About These Terms?</h2>
              <p className="text-white/70 mb-6">
                We're happy to clarify anything that's not crystal clear. No legal jargon in our responses!
              </p>

              <div className="grid md:grid-cols-3 gap-4 max-w-2xl mx-auto">
                <motion.a
                  href="mailto:legal@matchledger.in"
                  className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-3 rounded-lg font-medium transition-all duration-300 flex items-center justify-center space-x-2"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <span>Legal Team</span>
                </motion.a>

                <motion.a
                  href="mailto:support@matchledger.in"
                  className="bg-white/10 hover:bg-white/20 text-white px-6 py-3 rounded-lg font-medium border border-white/20 transition-all duration-300 flex items-center justify-center"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <span>Support</span>
                </motion.a>

                <Link to="/privacy-policy">
                  <motion.button
                    className="w-full bg-purple-500 hover:bg-purple-600 text-white px-6 py-3 rounded-lg font-medium transition-all duration-300"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <span>Privacy Policy</span>
                  </motion.button>
                </Link>
              </div>
            </div>
          </motion.div>

          {/* Footer */}
          <motion.div
            className="bg-white/5 backdrop-blur-xl rounded-2xl border border-white/10 p-6 mt-6 text-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.9 }}
          >
            <div className="text-sm text-white/60 space-y-2">
              <p>
                <strong>MatchLedger AI Terms of Service</strong> • Version 2.1 • Effective {new Date().toLocaleDateString()}
              </p>
              <p>
                We'll email you about any significant changes. Continued use means you accept the updated terms.
              </p>
              <p className="pt-2 border-t border-white/10">
                © 2024 MatchLedger Inc. • Built with ❤️ for better financial reconciliation
              </p>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
};

export default TermsOfService;