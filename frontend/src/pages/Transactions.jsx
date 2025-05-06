import React, { useState } from "react";
import { useBankData } from "../context/BankDataContext";

const Transactions = () => {
  const { accounts, transactions, getAccountName } = useBankData();

  const [filters, setFilters] = useState({
    account: "All",
    category: "All",
    dateRange: "all",
  });

  // Get unique categories for filter dropdown
  const categories = ["All", ...new Set(transactions.map((tx) => tx.category))];

  // Apply filters
  const filteredTransactions = transactions.filter((tx) => {
    // Filter by account
    if (
      filters.account !== "All" &&
      tx.accountId !== parseInt(filters.account)
    ) {
      return false;
    }

    // Filter by category
    if (filters.category !== "All" && tx.category !== filters.category) {
      return false;
    }

    // Filter by date range
    if (filters.dateRange !== "all") {
      const txDate = new Date(tx.date);
      const today = new Date();

      if (filters.dateRange === "week") {
        const oneWeekAgo = new Date();
        oneWeekAgo.setDate(today.getDate() - 7);
        if (txDate < oneWeekAgo) {
          return false;
        }
      } else if (filters.dateRange === "month") {
        const oneMonthAgo = new Date();
        oneMonthAgo.setMonth(today.getMonth() - 1);
        if (txDate < oneMonthAgo) {
          return false;
        }
      }
    }

    return true;
  });

  // Sort transactions by date (newest first)
  filteredTransactions.sort((a, b) => new Date(b.date) - new Date(a.date));

  // Calculate totals
  const totalIncome = filteredTransactions
    .filter((tx) => tx.amount > 0)
    .reduce((sum, tx) => sum + tx.amount, 0);

  const totalExpenses = filteredTransactions
    .filter((tx) => tx.amount < 0)
    .reduce((sum, tx) => sum + Math.abs(tx.amount), 0);

  // Handle filter changes
  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    setFilters((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h2 className="text-2xl font-semibold text-primary mb-6">
        Transaction History
      </h2>

      {/* Filters */}
      <div className="bg-white p-4 rounded-lg shadow-md mb-6 border border-background">
        <h3 className="text-lg font-medium text-primary mb-3">Filters</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Account Filter */}
          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1">
              Account
            </label>
            <select
              data-id="dropdown-account-filter"
              data-type="dropdown"
              data-label="Account Filter"
              className="w-full border border-background rounded-md p-2"
              name="account"
              value={filters.account}
              onChange={handleFilterChange}
            >
              <option value="All">All Accounts</option>
              {accounts.map((acc) => (
                <option key={acc.id} value={acc.id}>
                  {acc.type} ({acc.number})
                </option>
              ))}
            </select>
          </div>

          {/* Category Filter */}
          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1">
              Category
            </label>
            <select
              data-id="dropdown-category-filter"
              data-type="dropdown"
              data-label="Category Filter"
              className="w-full border border-background rounded-md p-2"
              name="category"
              value={filters.category}
              onChange={handleFilterChange}
            >
              {categories.map((category) => (
                <option key={category} value={category}>
                  {category}
                </option>
              ))}
            </select>
          </div>

          {/* Date Range Filter */}
          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1">
              Date Range
            </label>
            <select
              data-id="dropdown-date-filter"
              data-type="dropdown"
              data-label="Date Range Filter"
              className="w-full border border-background rounded-md p-2"
              name="dateRange"
              value={filters.dateRange}
              onChange={handleFilterChange}
            >
              <option value="all">All Time</option>
              <option value="week">Last 7 Days</option>
              <option value="month">Last 30 Days</option>
            </select>
          </div>
        </div>
      </div>

      {/* Transaction Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <div
          data-id="card-total-transactions"
          data-type="card"
          data-label="Total Transactions Card"
          className="bg-white shadow-md rounded-lg p-4 border border-background"
        >
          <h3 className="text-lg font-semibold text-primary">
            Total Transactions
          </h3>
          <p className="text-xl font-bold text-accent mt-2">
            {filteredTransactions.length}
          </p>
        </div>

        <div
          data-id="card-total-income"
          data-type="card"
          data-label="Total Income Card"
          className="bg-white shadow-md rounded-lg p-4 border border-background"
        >
          <h3 className="text-lg font-semibold text-primary">Total Income</h3>
          <p className="text-xl font-bold text-secondary mt-2">
            $ {totalIncome.toFixed(2)}
          </p>
        </div>

        <div
          data-id="card-total-expenses"
          data-type="card"
          data-label="Total Expenses Card"
          className="bg-white shadow-md rounded-lg p-4 border border-background"
        >
          <h3 className="text-lg font-semibold text-primary">Total Expenses</h3>
          <p className="text-xl font-bold text-text-primary mt-2">
            $ {totalExpenses.toFixed(2)}
          </p>
        </div>
      </div>

      {/* Transactions Table */}
      <div
        data-id="table-transactions"
        data-type="table"
        data-label="Transactions Table"
        className="bg-white rounded-lg shadow-md overflow-hidden border border-background"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-xs text-text-secondary uppercase bg-background">
              <tr>
                <th className="px-6 py-3">Date</th>
                <th className="px-6 py-3">Description</th>
                <th className="px-6 py-3">Account</th>
                <th className="px-6 py-3">Category</th>
                <th className="px-6 py-3 text-right">Amount</th>
              </tr>
            </thead>
            <tbody>
              {filteredTransactions.length > 0 ? (
                filteredTransactions.map((transaction) => (
                  <tr
                    key={transaction.id}
                    data-id={`row-tx-${transaction.id}`}
                    className="bg-white border-b hover:bg-background"
                  >
                    <td className="px-6 py-4">{transaction.date}</td>
                    <td className="px-6 py-4">{transaction.description}</td>
                    <td className="px-6 py-4">
                      {getAccountName(transaction.accountId)}
                    </td>
                    <td className="px-6 py-4">
                      <span className="bg-accent/10 text-accent text-xs font-medium px-2.5 py-0.5 rounded">
                        {transaction.category}
                      </span>
                    </td>
                    <td
                      className={`px-6 py-4 text-right font-medium ${
                        transaction.amount >= 0
                          ? "text-secondary"
                          : "text-text-primary"
                      }`}
                    >
                      {transaction.amount >= 0 ? "+" : "-"} $
                      {Math.abs(transaction.amount).toFixed(2)}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td
                    colSpan="5"
                    className="px-6 py-4 text-center text-text-secondary"
                  >
                    No transactions found matching your filter criteria.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Transactions;
