import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useBankData } from "../context/BankDataContext";

const Accounts = () => {
  const { accounts, transactions } = useBankData();
  const navigate = useNavigate();

  const [selectedAccount, setSelectedAccount] = useState("All");
  const [showAccountDetails, setShowAccountDetails] = useState(null);

  // Calculate total balance across all accounts
  const totalBalance = accounts.reduce(
    (sum, account) => sum + account.balance,
    0
  );

  const filteredAccounts =
    selectedAccount === "All"
      ? accounts
      : accounts.filter((acc) => acc.type === selectedAccount);

  const handleViewStatement = (accountId) => {
    // For now, just toggle the detail view within this component
    setShowAccountDetails(showAccountDetails === accountId ? null : accountId);
  };

  // Get account and transactions for the selected account detail
  const selectedAccountData = accounts.find(
    (acc) => acc.id === showAccountDetails
  );
  const accountTransactions = showAccountDetails
    ? transactions
        .filter((tx) => tx.accountId === showAccountDetails)
        .sort((a, b) => new Date(b.date) - new Date(a.date))
        .slice(0, 5)
    : [];

  return (
    <div className="p-6 max-w-5xl mx-auto">
      {/* Dashboard Summary */}
      {!showAccountDetails && (
        <div className="bg-background rounded-lg p-4 mb-6 border border-secondary/20">
          <h2 className="text-xl font-semibold text-primary mb-2">
            Financial Summary
          </h2>
          <div className="flex flex-wrap gap-4">
            <div>
              <p className="text-sm text-secondary">Total Balance</p>
              <p className="text-xl font-bold text-primary">
                ${totalBalance.toLocaleString()}
              </p>
            </div>
            <div>
              <p className="text-sm text-secondary">Total Accounts</p>
              <p className="text-xl font-bold text-primary">
                {accounts.length}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Account Detail View */}
      {showAccountDetails ? (
        <div>
          <button
            onClick={() => setShowAccountDetails(null)}
            className="mb-4 px-3 py-1 bg-background text-text-secondary rounded hover:bg-secondary/10 flex items-center text-sm"
          >
            ← Back to All Accounts
          </button>

          <div className="bg-white shadow-md rounded-lg p-6 border border-background mb-6">
            <h2 className="text-2xl font-semibold text-primary mb-4">
              {selectedAccountData.type} Account
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div>
                <p className="text-sm text-text-secondary">Account Number</p>
                <p className="text-lg font-medium text-text-primary">
                  {selectedAccountData.number}
                </p>
              </div>
              <div>
                <p className="text-sm text-text-secondary">Current Balance</p>
                <p className="text-lg font-bold text-secondary">
                  {selectedAccountData.currency}{" "}
                  {selectedAccountData.balance.toLocaleString()}
                </p>
              </div>
              <div>
                <p className="text-sm text-text-secondary">Status</p>
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-secondary/10 text-secondary">
                  Active
                </span>
              </div>
            </div>

            <h3 className="text-lg font-semibold text-primary mb-3">
              Recent Transactions
            </h3>
            {accountTransactions.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm text-left">
                  <thead className="text-xs text-text-secondary uppercase bg-background">
                    <tr>
                      <th className="px-4 py-2">Date</th>
                      <th className="px-4 py-2">Description</th>
                      <th className="px-4 py-2">Category</th>
                      <th className="px-4 py-2 text-right">Amount</th>
                    </tr>
                  </thead>
                  <tbody>
                    {accountTransactions.map((transaction) => (
                      <tr
                        key={transaction.id}
                        className="bg-white border-b hover:bg-background"
                      >
                        <td className="px-4 py-2">{transaction.date}</td>
                        <td className="px-4 py-2">{transaction.description}</td>
                        <td className="px-4 py-2">
                          <span className="bg-accent/10 text-accent text-xs font-medium px-2 py-0.5 rounded">
                            {transaction.category}
                          </span>
                        </td>
                        <td
                          className={`px-4 py-2 text-right font-medium ${
                            transaction.amount >= 0
                              ? "text-secondary"
                              : "text-text-primary"
                          }`}
                        >
                          {transaction.amount >= 0 ? "+" : "-"} $
                          {Math.abs(transaction.amount).toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-text-secondary">
                No recent transactions found for this account.
              </p>
            )}

            <div className="mt-4">
              <button
                onClick={() => navigate("/transactions")}
                className="text-accent hover:text-accent/80 text-sm font-medium"
              >
                View All Transactions →
              </button>
            </div>
          </div>
        </div>
      ) : (
        <>
          <h2 className="text-2xl font-semibold text-primary mb-4">
            Account Overview
          </h2>

          <div className="mb-6">
            <label className="block text-sm font-medium text-text-secondary mb-1">
              Filter by Account Type
            </label>
            <select
              data-id="dropdown-account-filter"
              data-type="dropdown"
              data-label="Account Type Filter"
              className="w-64 border border-secondary/20 p-2"
              value={selectedAccount}
              onChange={(e) => setSelectedAccount(e.target.value)}
            >
              <option value="All">All</option>
              <option value="Checking">Checking</option>
              <option value="Savings">Savings</option>
              <option value="Credit Card">Credit Card</option>
            </select>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {filteredAccounts.map((acc) => (
              <div
                key={acc.id}
                data-id={`card-${acc.id}`}
                data-type="card"
                data-label={`${acc.type} Account Card`}
                className="bg-white shadow-md rounded-lg p-4 border border-background"
              >
                <h3 className="text-lg font-semibold text-primary">
                  {acc.type}
                </h3>
                <p className="text-sm text-text-secondary mt-1">
                  Account No: {acc.number}
                </p>
                <p className="text-xl font-bold text-secondary mt-4">
                  {acc.currency} {acc.balance.toLocaleString()}
                </p>
                <button
                  data-id={`btn-view-${acc.id}`}
                  data-type="button"
                  data-label="View Statement"
                  className="mt-4 px-4 py-2 bg-accent text-white rounded hover:bg-accent/80"
                  onClick={() => handleViewStatement(acc.id)}
                >
                  View Statement
                </button>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
};

export default Accounts;
