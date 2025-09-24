import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';

// Smart Back Navigation Hook
const useSmartNavigation = () => {
  const { user } = useAuth();

  // Check if user has beta access or is admin
  const hasBetaAccess = user && (user.is_beta_user || user.isAdmin || user.role === 'admin');

  if (hasBetaAccess) {
    return { backPath: '/dashboard', backText: 'Back to Dashboard' };
  } else {
    return { backPath: '/beta', backText: 'Back to Beta' };
  }
};

const Contact = () => {
  const [activeCategory, setActiveCategory] = useState('getting-started');
  const [openFAQ, setOpenFAQ] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const { backPath, backText } = useSmartNavigation();

  const categories = [
    { id: 'getting-started', title: 'Getting Started', icon: '🚀', count: 6 },
    { id: 'privacy-security', title: 'Privacy & Security', icon: '🔒', count: 8 },
    { id: 'file-upload', title: 'File Upload & Processing', icon: '📁', count: 7 },
    { id: 'results', title: 'Understanding Results', icon: '📊', count: 5 },
    { id: 'account', title: 'Account Management', icon: '👤', count: 4 },
    { id: 'beta', title: 'Beta Program', icon: '🧪', count: 5 },
    { id: 'technical', title: 'Technical Issues', icon: '🔧', count: 6 }
  ];

  const faqs = {
    'getting-started': [
      {
        q: "What is MatchLedger AI and what does it do?",
        a: "MatchLedger AI is an intelligent financial reconciliation tool that automatically matches transactions between different financial records (like bank statements, credit card statements, and accounting records). It uses AI to identify matches, highlight discrepancies, and save you hours of manual work.",
        tags: ["basics", "overview"]
      },
      {
        q: "How do I get started with MatchLedger AI?",
        a: "Simply sign up for a free beta account, upload your financial files (CSV, Excel, or PDF formats), and let our AI do the matching. The whole process takes just a few minutes. We'll guide you through each step with helpful tooltips and examples.",
        tags: ["signup", "first-time"]
      },
      {
        q: "What file formats are supported?",
        a: "We support CSV, Excel (.xlsx, .xls), and PDF files. Most bank statements and accounting exports work perfectly. If you have a different format, contact support - we're always adding new format support.",
        tags: ["files", "formats"]
      },
      {
        q: "Do I need to install any software?",
        a: "No! MatchLedger AI runs completely in your web browser. It works on any device - desktop, tablet, or mobile. No downloads, no installation, no IT hassles.",
        tags: ["installation", "browser"]
      },
      {
        q: "How long does reconciliation take?",
        a: "Most reconciliations complete in 30 seconds to 2 minutes, depending on file size. Our AI processes thousands of transactions per minute. You'll see a progress bar and get notified when it's done.",
        tags: ["speed", "processing"]
      },
      {
        q: "Can I try it for free?",
        a: "Yes! The entire beta program is completely free. No credit card required, no hidden fees. We want you to test everything and give us feedback before we launch the paid version.",
        tags: ["free", "pricing", "beta"]
      }
    ],
    'privacy-security': [
      {
        q: "Is my financial data safe with MatchLedger?",
        a: "Absolutely. We use bank-level encryption (AES-256), secure cloud hosting on AWS, and automatically delete your files within 24 hours. We never store your financial data permanently and never share it with third parties.",
        tags: ["security", "encryption", "safety"]
      },
      {
        q: "How long do you keep my uploaded files?",
        a: "Your uploaded files are automatically deleted within 24 hours of processing. Reconciliation results are kept for 30 days so you can download them, then permanently deleted. You can request immediate deletion anytime.",
        tags: ["retention", "deletion", "storage"]
      },
      {
        q: "Who has access to my data?",
        a: "Only you and our automated AI systems. Our human staff never see your actual financial data. Even our developers and support team can only see anonymized system logs, never your bank statements or transaction details.",
        tags: ["access", "staff", "privacy"]
      },
      {
        q: "Do you sell my data to advertisers?",
        a: "Never. We don't sell, share, or monetize your personal data in any way. We don't have advertising partners, data brokers, or marketing companies accessing your information. Your data is yours.",
        tags: ["selling", "advertising", "monetization"]
      },
      {
        q: "What happens if there's a data breach?",
        a: "We have multiple security layers to prevent breaches, but if one occurred, we'd notify you within 24 hours, provide full details of what happened, and take immediate action to secure your data. We also carry cyber insurance.",
        tags: ["breach", "notification", "response"]
      },
      {
        q: "Can I delete all my data immediately?",
        a: "Yes! Go to Account Settings → Privacy → Delete All Data. We'll permanently remove everything within 24 hours and send you a confirmation email. No questions asked, no retention period.",
        tags: ["deletion", "account", "privacy"]
      },
      {
        q: "Are you GDPR compliant?",
        a: "Yes, we're fully GDPR compliant and follow strict European privacy standards. You have full control over your data, can request reports of what we store, and can delete everything anytime.",
        tags: ["gdpr", "compliance", "european"]
      },
      {
        q: "Do you use my data to train your AI?",
        a: "We use anonymous, aggregated patterns to improve our matching algorithms (like 'bank fees are usually negative amounts'), but we never use your specific transaction details, names, or account numbers for training.",
        tags: ["ai training", "machine learning", "anonymization"]
      }
    ],
    'file-upload': [
      {
        q: "What's the maximum file size I can upload?",
        a: "Currently 50MB per file during beta. This handles most bank statements and accounting exports. Need to upload larger files? Contact support - we can increase your limit.",
        tags: ["file size", "limits", "upload"]
      },
      {
        q: "Why won't my file upload?",
        a: "Common issues: file too large (50MB max), unsupported format, or corrupted file. Try: 1) Check file size, 2) Save as CSV if it's Excel, 3) Clear your browser cache, 4) Try a different browser. Still stuck? Email us the error message.",
        tags: ["upload issues", "troubleshooting", "errors"]
      },
      {
        q: "Can I upload password-protected files?",
        a: "Not yet, but it's coming soon! For now, remove password protection before uploading. We're working on secure password handling that never stores your passwords.",
        tags: ["passwords", "protected files", "security"]
      },
      {
        q: "How do I prepare my files for best results?",
        a: "Best practices: 1) Include transaction dates, amounts, and descriptions, 2) Remove extra headers or footers, 3) Use consistent date formats, 4) Save Excel files as CSV when possible. Our AI is smart but clean data = better results.",
        tags: ["preparation", "best practices", "optimization"]
      },
      {
        q: "Can I upload multiple files at once?",
        a: "Yes! Drag and drop multiple files or use Ctrl+click to select multiple files. We'll process them in order and show progress for each one. Great for monthly statements or multiple accounts.",
        tags: ["multiple files", "batch upload", "efficiency"]
      },
      {
        q: "What if my bank statement is in PDF format?",
        a: "We can extract data from most PDF bank statements! Upload them just like any other file. If the PDF is image-based or has unusual formatting, we might not catch everything - let us know if you have issues.",
        tags: ["pdf", "bank statements", "extraction"]
      },
      {
        q: "Do you support international currencies and formats?",
        a: "Yes! We support major currencies (USD, EUR, GBP, INR, etc.) and international date formats (DD/MM/YYYY, MM/DD/YYYY, etc.). Our AI automatically detects formats and currencies in your files.",
        tags: ["international", "currencies", "formats"]
      },
      {
        q: "Can I upload files from mobile devices?",
        a: "Absolutely! The upload works great on phones and tablets. You can upload directly from your phone's files, cloud storage apps, or even take photos of paper statements (though typed files work better).",
        tags: ["mobile", "phone", "tablet"]
      }
    ],
    'results': [
      {
        q: "How do I read my reconciliation results?",
        a: "Results show three sections: 1) Perfect Matches (green) - transactions that match exactly, 2) Potential Matches (yellow) - similar transactions that might match, 3) Unmatched (red) - transactions with no clear match. Click any item for details.",
        tags: ["results", "interpretation", "matches"]
      },
      {
        q: "What does 'confidence score' mean?",
        a: "It's how sure our AI is about a match, from 0-100%. Above 90% = very confident, 70-90% = likely match, below 70% = uncertain. Use your judgment for lower scores - you know your business best.",
        tags: ["confidence", "scoring", "accuracy"]
      },
      {
        q: "Can I manually confirm or reject matches?",
        a: "Yes! Click any potential match to accept or reject it. You can also add notes explaining why you made that decision. This helps train our AI to match your preferences better next time.",
        tags: ["manual review", "confirm", "reject"]
      },
      {
        q: "How do I export my results?",
        a: "Click the 'Export Results' button to download as Excel, CSV, or PDF. The export includes all matches, unmatched items, your manual decisions, and a summary report. Perfect for your accounting records.",
        tags: ["export", "download", "reporting"]
      },
      {
        q: "What should I do with unmatched transactions?",
        a: "Unmatched items might be: 1) Timing differences (transaction posted on different dates), 2) Missing from one file, 3) Data entry errors, or 4) Genuine discrepancies. Investigate each one - they often reveal important issues.",
        tags: ["unmatched", "discrepancies", "investigation"]
      }
    ],
    'account': [
      {
        q: "How do I change my email address?",
        a: "Go to Account Settings → Profile → Change Email. Enter your new email and verify it with the code we send. Your reconciliation history stays with your account.",
        tags: ["email", "account", "profile"]
      },
      {
        q: "How do I delete my account?",
        a: "Account Settings → Privacy → Delete Account. This permanently removes all your data within 24 hours. Download any important results first - this action can't be undone.",
        tags: ["delete account", "data removal", "permanent"]
      },
      {
        q: "Can I share my account with team members?",
        a: "Account sharing isn't available yet, but team features are coming! For now, each person needs their own free beta account. We're building proper team collaboration for the full launch.",
        tags: ["sharing", "team", "collaboration"]
      },
      {
        q: "I forgot my password. How do I reset it?",
        a: "Click 'Forgot Password' on the login page. We'll email you a reset link that's valid for 1 hour. If you don't get the email, check your spam folder or contact support.",
        tags: ["password", "reset", "login"]
      }
    ],
    'beta': [
      {
        q: "How long will the beta program last?",
        a: "The beta runs until mid-2024. We'll give all beta users at least 30 days notice before it ends, plus special early-bird pricing when we launch the paid version.",
        tags: ["beta duration", "timeline", "launch"]
      },
      {
        q: "What happens to my data when beta ends?",
        a: "You can export everything before beta ends. If you don't upgrade to the paid version, we'll delete all data 30 days after beta ends (with multiple reminder emails).",
        tags: ["beta end", "data migration", "transition"]
      },
      {
        q: "Will the service always be free?",
        a: "The beta is free, but we'll charge for the full version (fair pricing based on usage). Beta users get lifetime discounts and first access to new features. We'll never surprise you with sudden charges.",
        tags: ["pricing", "free", "future costs"]
      },
      {
        q: "How can I provide feedback?",
        a: "We love feedback! Use the feedback button in the app, email feedback@matchledger.in, or join our monthly user calls. Your input directly shapes the product roadmap.",
        tags: ["feedback", "suggestions", "improvement"]
      },
      {
        q: "What features are you adding next?",
        a: "Coming soon: team collaboration, API access, automated scheduled reconciliations, integration with QuickBooks/Xero, and mobile apps. Beta users vote on feature priorities!",
        tags: ["roadmap", "features", "development"]
      }
    ],
    'technical': [
      {
        q: "Which browsers work best?",
        a: "Chrome, Firefox, Safari, and Edge all work great. For best performance, use the latest version with JavaScript enabled. Internet Explorer isn't supported (it's too old!).",
        tags: ["browsers", "compatibility", "requirements"]
      },
      {
        q: "Why is the site running slowly?",
        a: "Try: 1) Refresh the page, 2) Clear browser cache, 3) Close other tabs, 4) Check your internet connection. If it's still slow, email us - we monitor performance closely.",
        tags: ["performance", "speed", "troubleshooting"]
      },
      {
        q: "I'm getting an error message. What do I do?",
        a: "Take a screenshot of the error and email it to support@matchledger.in with details about what you were doing. We usually respond within a few hours and fix bugs quickly.",
        tags: ["errors", "bugs", "support"]
      },
      {
        q: "Can I use MatchLedger on my phone?",
        a: "Yes! The website works on all devices. For the best mobile experience, use your phone's browser in landscape mode when reviewing results.",
        tags: ["mobile", "responsive", "phone"]
      },
      {
        q: "Do you have an API?",
        a: "Not yet, but it's high on our roadmap! We're building a REST API that will let you integrate MatchLedger into your existing systems. Beta users get early access.",
        tags: ["api", "integration", "development"]
      },
      {
        q: "Is there a desktop app?",
        a: "The web app works so well that we haven't built a desktop version yet. It runs offline once loaded and feels just like a native app. Desktop apps might come later based on user demand.",
        tags: ["desktop", "app", "native"]
      }
    ]
  };

  const contactOptions = [
    {
      title: "Email Support",
      description: "Get help with technical issues, account problems, or general questions",
      email: "support@matchledger.in",
      responseTime: "Usually within 4 hours",
      icon: "📧",
      color: "blue"
    },
    {
      title: "Privacy Questions",
      description: "Data protection, privacy concerns, or GDPR requests",
      email: "privacy@matchledger.in",
      responseTime: "Within 24 hours",
      icon: "🔒",
      color: "green"
    },
    {
      title: "Legal & Terms",
      description: "Questions about terms of service, licensing, or legal matters",
      email: "legal@matchledger.in",
      responseTime: "Within 48 hours",
      icon: "⚖️",
      color: "purple"
    },
    {
      title: "Product Feedback",
      description: "Feature requests, bug reports, or suggestions for improvement",
      email: "feedback@matchledger.in",
      responseTime: "We read every message",
      icon: "💡",
      color: "orange"
    }
  ];

  const colorClasses = {
    blue: "from-blue-500/10 to-blue-600/10 border-blue-400/20 text-blue-300",
    green: "from-green-500/10 to-green-600/10 border-green-400/20 text-green-300",
    purple: "from-purple-500/10 to-purple-600/10 border-purple-400/20 text-purple-300",
    orange: "from-orange-500/10 to-orange-600/10 border-orange-400/20 text-orange-300"
  };

  const filteredFAQs = faqs[activeCategory]?.filter(faq =>
    faq.q.toLowerCase().includes(searchTerm.toLowerCase()) ||
    faq.a.toLowerCase().includes(searchTerm.toLowerCase()) ||
    faq.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()))
  ) || [];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-indigo-900 to-slate-900 text-white">
      {/* Background Animation */}
      <div className="absolute inset-0 overflow-hidden">
        {[...Array(30)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-indigo-400/20 rounded-full"
            animate={{
              x: [Math.random() * 1200, Math.random() * 1200],
              y: [Math.random() * 800, Math.random() * 800],
            }}
            transition={{
              duration: Math.random() * 20 + 10,
              repeat: Infinity,
              repeatType: "reverse",
            }}
            initial={{
              x: Math.random() * 1200,
              y: Math.random() * 800,
            }}
          />
        ))}
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          {/* Header */}
          <div className="text-center mb-12">
            <Link
              to={backPath}
              className="inline-flex items-center text-indigo-300 hover:text-white mb-6 transition-colors group"
            >
              <svg className="w-5 h-5 mr-2 transition-transform group-hover:-translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7"></path>
              </svg>
              {backText}
            </Link>

            <motion.h1
              className="text-4xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-white via-indigo-200 to-purple-200 bg-clip-text text-transparent"
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              Help Center
            </motion.h1>

            <motion.p
              className="text-xl text-white/70 max-w-3xl mx-auto mb-8"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
            >
              Find answers to common questions or get personalized support from our team
            </motion.p>

            {/* Search Bar */}
            <motion.div
              className="max-w-md mx-auto"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              <div className="relative">
                <svg className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-white/50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                </svg>
                <input
                  type="text"
                  placeholder="Search FAQs..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-10 pr-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/50 focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-400/20"
                />
              </div>
            </motion.div>
          </div>

          <div className="grid lg:grid-cols-4 gap-8">
            {/* Category Sidebar */}
            <div className="lg:col-span-1">
              <div className="sticky top-8">
                <h3 className="text-lg font-semibold mb-4 text-indigo-300">Browse by Category</h3>
                <nav className="space-y-2">
                  {categories.map((category, index) => (
                    <motion.button
                      key={category.id}
                      onClick={() => {
                        setActiveCategory(category.id);
                        setSearchTerm('');
                        setOpenFAQ(null);
                      }}
                      className={`w-full text-left p-3 rounded-lg transition-all duration-300 flex items-center justify-between ${
                        activeCategory === category.id
                          ? 'bg-indigo-500/20 border border-indigo-400/30 text-white'
                          : 'bg-white/5 hover:bg-white/10 text-white/70 hover:text-white border border-transparent'
                      }`}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <div className="flex items-center space-x-3">
                        <span className="text-lg">{category.icon}</span>
                        <span className="font-medium">{category.title}</span>
                      </div>
                      <span className="bg-white/20 text-xs px-2 py-1 rounded-full">
                        {category.count}
                      </span>
                    </motion.button>
                  ))}
                </nav>
              </div>
            </div>

            {/* FAQ Content */}
            <div className="lg:col-span-3">
              {/* Category Header */}
              <div className="mb-6">
                <div className="flex items-center space-x-3 mb-2">
                  <span className="text-2xl">
                    {categories.find(cat => cat.id === activeCategory)?.icon}
                  </span>
                  <h2 className="text-2xl font-bold text-white">
                    {categories.find(cat => cat.id === activeCategory)?.title}
                  </h2>
                </div>
                <p className="text-white/60">
                  {filteredFAQs.length} question{filteredFAQs.length !== 1 ? 's' : ''}
                  {searchTerm && ` matching "${searchTerm}"`}
                </p>
              </div>

              {/* FAQ List */}
              <div className="space-y-4">
                {filteredFAQs.length === 0 ? (
                  <motion.div
                    className="text-center py-12"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                  >
                    <div className="text-6xl mb-4">🔍</div>
                    <h3 className="text-xl font-semibold text-white mb-2">No results found</h3>
                    <p className="text-white/60 mb-4">
                      Try a different search term or browse other categories
                    </p>
                    <button
                      onClick={() => setSearchTerm('')}
                      className="text-indigo-400 hover:text-white transition-colors"
                    >
                      Clear search
                    </button>
                  </motion.div>
                ) : (
                  filteredFAQs.map((faq, index) => (
                    <motion.div
                      key={index}
                      className="bg-white/5 backdrop-blur-xl rounded-xl border border-white/10 overflow-hidden"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 }}
                    >
                      <motion.button
                        onClick={() => setOpenFAQ(openFAQ === `${activeCategory}-${index}` ? null : `${activeCategory}-${index}`)}
                        className="w-full p-4 text-left hover:bg-white/5 transition-colors duration-300"
                        whileHover={{ scale: 1.01 }}
                        whileTap={{ scale: 0.99 }}
                      >
                        <div className="flex items-center justify-between">
                          <h3 className="text-lg font-medium text-white pr-4">{faq.q}</h3>
                          <motion.svg
                            className={`w-5 h-5 text-white/50 flex-shrink-0 transition-transform duration-300`}
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                            animate={{ rotate: openFAQ === `${activeCategory}-${index}` ? 180 : 0 }}
                          >
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                          </motion.svg>
                        </div>
                      </motion.button>

                      <motion.div
                        className="overflow-hidden"
                        initial={false}
                        animate={{
                          height: openFAQ === `${activeCategory}-${index}` ? 'auto' : 0,
                          opacity: openFAQ === `${activeCategory}-${index}` ? 1 : 0
                        }}
                        transition={{ duration: 0.3, ease: "easeInOut" }}
                      >
                        <div className="p-4 pt-0 border-t border-white/10">
                          <div className="text-white/80 leading-relaxed mb-3">
                            {faq.a}
                          </div>
                          <div className="flex flex-wrap gap-2">
                            {faq.tags.map((tag, tagIndex) => (
                              <span
                                key={tagIndex}
                                className="bg-indigo-500/20 text-indigo-300 px-2 py-1 rounded text-xs"
                              >
                                {tag}
                              </span>
                            ))}
                          </div>
                        </div>
                      </motion.div>
                    </motion.div>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Contact Support Section */}
          <motion.div
            className="mt-16"
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
          >
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-white mb-4">Still Need Help?</h2>
              <p className="text-white/70 text-lg">
                Our support team is here to help. Choose the best way to reach us:
              </p>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              {contactOptions.map((option, index) => (
                <motion.div
                  key={index}
                  className={`bg-gradient-to-br ${colorClasses[option.color]} backdrop-blur-xl rounded-xl border p-6 text-center hover:scale-105 transition-all duration-300`}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 1 + index * 0.1 }}
                  whileHover={{ y: -5 }}
                >
                  <div className="text-4xl mb-4">{option.icon}</div>
                  <h3 className="text-lg font-semibold text-white mb-2">{option.title}</h3>
                  <p className="text-white/70 text-sm mb-4">{option.description}</p>
                  
                  {/* Email as prominent button */}
                  <motion.a
                    href={`mailto:${option.email.trim()}`}
                    onClick={(e) => {
                      e.preventDefault();
                      const cleanEmail = option.email.replace(/[<>]/g, '').trim();
                      window.location.href = `mailto:${cleanEmail}`;

                    }}
                      
                    className={`inline-block px-6 py-3 rounded-lg font-medium text-lg transition-all duration-300 ${
                      option.color === 'blue' ? 'bg-blue-500 hover:bg-blue-600' : 
                      option.color === 'green' ? 'bg-green-500 hover:bg-green-600' : 
                      option.color === 'purple' ? 'bg-purple-500 hover:bg-purple-600' : 
                      'bg-orange-500 hover:bg-orange-600'
                    } text-white mb-3 block`}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    {option.email.replace(/[<>]/g, '').trim()}
                  </motion.a>
                  
                  <p className="text-xs text-white/50">{option.responseTime}</p>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Quick Links */}
          <motion.div
            className="mt-12 bg-white/5 backdrop-blur-xl rounded-2xl border border-white/10 p-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.2 }}
          >
            <h3 className="text-xl font-semibold text-white mb-6 text-center">Quick Links</h3>
            <div className="grid md:grid-cols-3 gap-6 text-center">
              <motion.button
                onClick={() => window.open('/privacy-policy', '_blank')}
                className="p-4 bg-white/5 rounded-lg hover:bg-white/10 transition-colors border border-white/10 hover:border-white/20"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="text-2xl mb-2">🔒</div>
                <div className="font-medium text-white">Privacy Policy</div>
                <div className="text-white/60 text-sm">How we protect your data</div>
              </motion.button>

              <motion.button
                onClick={() => window.open('/terms-of-service', '_blank')}
                className="p-4 bg-white/5 rounded-lg hover:bg-white/10 transition-colors border border-white/10 hover:border-white/20"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="text-2xl mb-2">📋</div>
                <div className="font-medium text-white">Terms of Service</div>
                <div className="text-white/60 text-sm">Rules and agreements</div>
              </motion.button>

              <motion.a
                href="mailto:feedback@matchledger.in"
                className="block p-4 bg-white/5 rounded-lg hover:bg-white/10 transition-colors border border-white/10 hover:border-white/20"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="text-2xl mb-2">💡</div>
                <div className="font-medium text-white">Feature Requests</div>
                <div className="text-white/60 text-sm">Suggest improvements</div>
              </motion.a>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
};

export default Contact;